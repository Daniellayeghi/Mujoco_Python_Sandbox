import math
import random

import mujoco
import torch

from models import Cartpole, ModelParams
from neural_value_synthesis_diffeq import *
import matplotlib.pyplot as plt
from utilities.adahessian import AdaHessian
from utilities.mujoco_torch import torch_mj_set_attributes, SimulationParams, torch_mj_inv

sim_params = SimulationParams(6, 4, 2, 2, 2, 1, 1, 300)
cp_params = ModelParams(2, 2, 1, 4, 4)

prev_cost, diff, iteration, tol, max_iter, step_size = 0, 100.0, 1, 0, 3000, 1.0
Q = torch.diag(torch.Tensor([1, 2, 2, 0.01, 0.01])).repeat(sim_params.nsim, 1, 1).to(device)
R = torch.diag(torch.Tensor([0.1])).repeat(sim_params.nsim, 1, 1).to(device)
Qf = torch.diag(torch.Tensor([50000, 600000, 600000, 500, 6000])).repeat(sim_params.nsim, 1, 1).to(device)
cartpole = Cartpole(sim_params.nsim, cp_params)


def wrap_free_state(x: torch.Tensor):
    q, v = x[:, :, :sim_params.nq], x[:, :, sim_params.nq:]
    q_new = torch.cat((torch.sin(q[:, :, 1]), torch.cos(q[:, :, 1]) - 1, q[:, :, 0]), 1).unsqueeze(1)
    return torch.cat((q_new, v), 2)


def wrap_free_state_batch(x: torch.Tensor):
    q, v = x[:, :, :, :sim_params.nq], x[:, :, :, sim_params.nq:]
    q_new = torch.cat((torch.sin(q[:, :, :, 1]), torch.cos(q[:, :, :, 1]) - 1, q[:, :, :, 0]), 2).unsqueeze(2)
    return torch.cat((q_new, v), 3)


def bounded_state(x: torch.Tensor):
    qc, qp, qdc, qdp  = x[:, :, 0].clone().unsqueeze(1), x[:, :, 1].clone().unsqueeze(1), x[:, :, 2].clone().unsqueeze(1), x[:, :, 3].clone().unsqueeze(1)
    qp = (qp+2 * torch.pi)%torch.pi
    return torch.cat((qc, qp, qdc, qdp), 2)


def bounded_traj(x: torch.Tensor):
    def bound(angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    qc, qp, qdc, qdp = x[:, :, :, 0].clone().unsqueeze(2), x[:, :, :, 1].clone().unsqueeze(2), x[:, :, :, 2].clone().unsqueeze(2), x[:, :, :, 3].clone().unsqueeze(2)
    qp = bound(qp)
    return torch.cat((qc, qp, qdc, qdp), 3)


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


bad_init = False


def loss_func(x: torch.Tensor):
    x = wrap_free_state(x)
    return x @ Q @ x.mT

def backup_loss(x: torch.Tensor):
    t, nsim, r, c = x.shape
    x_final = x[-1:, :, :, :].view(1, nsim, r, c).clone()
    x_final_w = wrap_free_state_batch(x_final)
    l_running = torch.sum(x_final_w @ Q @ x_final_w.mT, 0).squeeze()
    value = nn_value_func(0, x_final).squeeze()
    return torch.mean(torch.square(value - l_running))


def batch_state_loss(x: torch.Tensor):
    x = wrap_free_state_batch(x)
    t, nsim, r, c = x.shape
    x_run = x[0:-1, :, :, :].view(t -1, nsim, r, c).clone()
    x_final = x[-1:, :, :, :].view(1, nsim, r, c).clone()
    l_running = torch.sum(x_run @ Q @ x_run.mT, 0).squeeze()
    l_terminal = (x_final @ Qf @ x_final.mT).squeeze()

    return torch.mean(l_running + l_terminal)


def batch_ctrl_loss(acc: torch.Tensor):
    qddc = acc[:, :, :, 0].unsqueeze(2).clone()
    l_ctrl = torch.sum(qddc @ R @ qddc.mT, 0).squeeze()
    return torch.mean(l_ctrl)


def loss_function(x, acc):
    return batch_ctrl_loss(acc) + batch_state_loss(x)


nn_value_func = NNValueFunction(sim_params.nqv).to(device)
dyn_system = ProjectedDynamicalSystem(nn_value_func, loss_func, sim_params, cartpole, mode='proj').to(device)
time = torch.linspace(0, (sim_params.ntime - 1) * 0.01, sim_params.ntime).to(device)
optimizer = torch.optim.Adam(dyn_system.parameters(), lr=1e-3)
# q_init = torch.Tensor([0, torch.pi]).repeat(sim_params.nsim, 1, 1).to(device)
qp_init = torch.FloatTensor(sim_params.nsim, 1, 1).uniform_(3.14, 3.14) * 1
qc_init = torch.FloatTensor(sim_params.nsim, 1, 1).uniform_(0, 0) * 1
qd_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nv).uniform_(0, 0) * 1
x_init = torch.cat((qc_init, qp_init, qd_init), 2).to(device)
bad_init = False

if __name__ == "__main__":
    while iteration < max_iter:
        if bad_init:
            nn_value_func = NNValueFunction(sim_params.nqv).to(device)
            dyn_system = ProjectedDynamicalSystem(nn_value_func, loss_func, sim_params, cartpole).to(device)

        optimizer.zero_grad()
        traj = odeint(dyn_system, x_init, time, method='euler', options=dict(step_size=0.02))
        acc = compose_acc(traj, 0.02)
        xxd = compose_xxd(traj, acc)
        loss = loss_function(traj, acc)
        bad_init = loss.isnan()
        loss.backward(create_graph=True)
        optimizer.step()

        # for param in dyn_system.parameters():
        #     print(f"\n{param}\n")

        print(f"Epochs: {iteration}, Loss: {loss.item()}, iteration: {iteration % 10}")

        selection = random.randint(0, sim_params.nsim - 1)

        if iteration % 1 == 0:
            fig_1 = plt.figure(1)
            for i in range(sim_params.nsim):
                # traj_b = bounded_traj(traj)
                qpole = traj[:, i, 0, 1].clone().detach()
                qdpole = traj[:, i, 0, 3].clone().detach()
                plt.plot(qpole, qdpole)

            plt.pause(0.001)
            fig_2 = plt.figure(2)
            ax_2 = plt.axes()
            plt.plot(traj[:, selection, 0, 0].clone().detach())
            plt.plot(qpole)
            plt.plot(acc[:, selection, 0, 0].clone().detach())
            plt.plot(acc[:, selection, 0, 1].clone().detach())
            plt.pause(0.001)
            ax_2.set_title(loss.item)
            fig_1.clf()
            fig_2.clf()

        if iteration % 10 == 0:
            from animations.cartpole import animate_cartpole
            cart = traj[:, selection, 0, 0].clone().detach().numpy()
            pole = traj[:, selection, 0, 1].clone().detach().numpy()
            animate_cartpole(cart, pole)

        iteration += 1
