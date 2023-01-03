import math
import mujoco
import torch

from models import Cartpole, ModelParams
from neural_value_synthesis_diffeq import *
import matplotlib.pyplot as plt
from utilities.torch_utils import save_models
from utilities.mujoco_torch import torch_mj_set_attributes, SimulationParams, torch_mj_inv

sim_params = SimulationParams(6, 4, 2, 2, 2, 1, 1, 100)
cp_params = ModelParams(2, 2, 1, 4, 4)

prev_cost, diff, iteration, tol, max_iter, step_size = 0, 100.0, 1, 0, 3000, 1.0
Q = torch.diag(torch.Tensor([2, 1, 0, 0])).repeat(sim_params.nsim, 1, 1).to(device)
R = torch.diag(torch.Tensor([0])).repeat(sim_params.nsim, 1, 1).to(device)
Qf = torch.diag(torch.Tensor([50000, 100000, 500, 500])).repeat(sim_params.nsim, 1, 1).to(device)
cartpole = Cartpole(1, cp_params)

def wrap_free_state(x: torch.Tensor):
    q, v = x[:, :sim_params.nq].unsqeeze(1), x[:, sim_params.nq:].unsqeeze(1)
    q_new = torch.Tensor([torch.sin(q[:, 0]), torch.cos(q[:, 0]), q[:, 1]])
    return torch.cat((q_new, v), 1)

def bounded_state(x: torch.Tensor):
    qc, qp, qdc, qdp  = x[:, :, 0].clone().unsqueeze(1), x[:, :, 1].clone().unsqueeze(1), x[:, :, 2].clone().unsqueeze(1), x[:, :, 3].clone().unsqueeze(1)
    qp = (qp+ 2 * torch.pi)%torch.pi
    return torch.cat((qc, qp, qdc, qdp), 2)

def bounded_traj(x: torch.Tensor):
    qc, qp, qdc, qdp  = x[:, :, :, 0].clone().unsqueeze(2), x[:, :, :, 1].clone().unsqueeze(2), x[:, :, :, 2].clone().unsqueeze(2), x[:, :, :, 3].clone().unsqueeze(2)
    qp = (qp+ 2 * torch.pi)%torch.pi
    return torch.cat((qc, qp, qdc, qdp), 3)


def loss_func(x: torch.Tensor):
    # x = bounded_state(x)
    return x @ Q @ x.mT


def batch_state_loss(x: torch.Tensor):
    t, nsim, r, c = x.shape
    # x = bounded_traj(x)
    x_run = x[0:-1, :, :, :].view(t - 1, nsim, r, c).clone()
    x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
    l_running = torch.sum(x_run @ Q @ x_run.mT, 0).squeeze()
    l_terminal = (x_final @ Qf @ x_final.mT).squeeze()

    return torch.mean(l_running + l_terminal)


def batch_ctrl_loss(acc: torch.Tensor):
    qddc = acc[:, :, :, 0].unsqueeze(2).clone()
    l_ctrl = torch.sum(qddc @ R @ qddc.mT, 0).squeeze()
    return torch.mean(l_ctrl)


def loss_function(x, acc):
    return batch_ctrl_loss(acc) + batch_state_loss(x)


class NNValueFunction(nn.Module):
    def __init__(self, n_in):
        super(NNValueFunction, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(n_in, 32, bias=False),
            nn.Softplus(),
            nn.Linear(32, 64, bias=False),
            nn.Softplus(),
            nn.Linear(64, 256, bias=False),
            nn.Softplus(),
            nn.Linear(256, 64, bias=False),
            nn.Softplus(),
            nn.Linear(64, 32, bias=False),
            nn.Softplus(),
            nn.Linear(32, 1, bias=False)
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
optimizer = torch.optim.Adam(dyn_system.parameters(), lr=3e-4, amsgrad=True)
# q_init = torch.Tensor([0, torch.pi]).repeat(sim_params.nsim, 1, 1).to(device)
q_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nq).uniform_(0.2, 1.5 * torch.pi) * 1
qd_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nv).uniform_(0, 0) * 1
x_init = torch.cat((q_init, qd_init), 2).to(device)
pos_arr = torch.linspace(0, 2*torch.pi, 100).to(device)
vel_arr = torch.linspace(-.2, .2, 100).to(device)
f_mat = torch.zeros((100, 100)).to(device)
[X, Y] = torch.meshgrid(pos_arr.squeeze().cpu(), vel_arr.squeeze().cpu())

if __name__ == "__main__":
    while iteration < max_iter:
        optimizer.zero_grad()
        traj = odeint(dyn_system, x_init, time, method='euler')
        acc = compose_acc(traj, 0.01)
        xxd = compose_xxd(traj, acc)
        loss = loss_function(traj, acc)

        # dyn_system.step *= step_size
        # print(f"Stepping with {dyn_system.step}")

        diff = math.fabs(prev_cost - loss.item())
        prev_cost = loss.item()
        loss.backward()
        optimizer.step()

        for param in dyn_system.parameters():
            print(f"\n{param}\n")

        print(f"Epochs: {iteration}, Loss: {loss.item()}, iteration: {iteration % 10}")
        plt.clf()
        plt.figure(1)
        qpole = traj[:, 0, 0, 1].clone().detach()
        qdpole = traj[:, 0, 0, 3].clone().detach()
        plt.plot(qpole, qdpole)
        plt.pause(0.001)
        plt.clf()
        plt.figure(2)
        traj_b = bounded_traj(traj)
        plt.plot(traj[:, 0, 0, 0].clone().detach())
        plt.plot(qpole)
        plt.plot(acc[:, 0, 0, 0].clone().detach())
        plt.plot(acc[:, 0, 0, 1].clone().detach())
        plt.pause(0.001)

        if iteration % 10 == 0:
            from animations.cartpole import animate_cartpole
            cart = traj[:, 0, 0, 0].clone().detach().numpy()
            pole = traj[:, 0, 0, 1].clone().detach().numpy()
            animate_cartpole(cart, pole)
            plt.clf()

        iteration += 1
