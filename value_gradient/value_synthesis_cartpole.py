import math
import mujoco
import torch

from models import Cartpole
from neural_value_synthesis_diffeq import *
from utilities.torch_utils import save_models
from utilities.mujoco_torch import torch_mj_set_attributes, SimulationParams, torch_mj_inv

sim_params = SimulationParams(6, 4, 2, 2, 2, 1, 50, 100)
prev_cost, diff, iteration, tol, max_iter, step_size = 0, 100.0, 1, 0.0005, 100, 1.01
Q = torch.diag(torch.Tensor([1, 1, 1, 1])).repeat(sim_params.nsim, 1, 1).to(device)
R = torch.diag(torch.Tensor([0.5])).repeat(sim_params.nsim, 1, 1).to(device)
Qf = torch.diag(torch.Tensor([1, 1, 1, 1])).repeat(sim_params.nsim, 1, 1).to(device)
cartpole = Cartpole(50)

def wrap_free_state(x: torch.Tensor):
    q, v = x[:, :sim_params.nq].unsqeeze(1), x[:, sim_params.nq:].unsqeeze(1)
    q_new = torch.Tensor([torch.sin(q[:, 0]), torch.cos(q[:, 0]), q[:, 1]])
    return torch.cat((q_new, v), 1)

def bounded_state(x: torch.Tensor):
    x[:, :sim_params.nq] = (x[:, :sim_params.nq] + 2 * torch.pi)%torch.pi
    return x.clone()


def loss_func(x: torch.Tensor):
    # x = bounded_state(x)
    return x @ Q @ x.mT


def batch_state_loss(x: torch.Tensor):
    t, nsim, r, c = x.shape
    # x = bounded_state(x)
    x_run = x[0:-1, :, :, :].view(t - 1, nsim, r, c).clone()
    x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
    l_running = torch.sum(x_run @ Q @ x_run.mT, 0).squeeze()
    l_terminal = (x_final @ Qf @ x_final.mT).squeeze()

    return torch.mean(l_running + l_terminal)



class NNValueFunction(nn.Module):
    def __init__(self, n_in):
        super(NNValueFunction, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(n_in, 8, bias=False),
            nn.Softplus(),
            nn.Linear(8, 4, bias=False),
            nn.Softplus(),
            nn.Linear(4, 1, bias=False)
        )

        def init_weights(net):
            if type(net) == nn.Linear:
                torch.nn.init.xavier_uniform(net.weight)

        self.nn.apply(init_weights)

    def forward(self, t, x):
        return self.nn(x)

nn_value_func = NNValueFunction(sim_params.nqv).to(device)
dyn_system = ProjectedDynamicalSystem(nn_value_func, loss_func, sim_params, cartpole).to(device)
time = torch.linspace(0, (sim_params.ntime - 1) * 0.01, sim_params.ntime).to(device)
optimizer = torch.optim.AdamW(dyn_system.parameters(), lr=3e-3)
q_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nq).uniform_(0, 1) * 3
qd_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nv).uniform_(-.2, .2) * 1
x_init = torch.cat((q_init, qd_init), 2).to(device)


if __name__ == "__main__":

    while diff > tol or iteration > max_iter:
        optimizer.zero_grad()
        traj = odeint(dyn_system, x_init, time)
        acc = compose_acc(traj, 0.01)
        xxd = compose_xxd(traj, acc)
        loss = batch_state_loss(traj)

        dyn_system.step *= step_size
        print(f"Stepping with {dyn_system.step}")

        diff = math.fabs(prev_cost - loss.item())
        prev_cost = loss.item()
        loss.backward()
        optimizer.step()

        for param in dyn_system.parameters():
            print(f"\n{param}\n")

        print(f"Epochs: {iteration}, Loss: {loss.item()}")
        iteration += 1