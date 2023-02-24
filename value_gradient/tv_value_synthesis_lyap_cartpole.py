import random
import numpy as np
import torch

from models import Cartpole, ModelParams
from animations.cartpole import init_fig_cp, animate_cartpole
from neural_value_synthesis_diffeq import *
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from utilities.mujoco_torch import SimulationParams
from PSDNets import PosDefICNN
sim_params = SimulationParams(6, 4, 2, 2, 2, 1, 200, 240, 0.008)
cp_params = ModelParams(2, 2, 1, 4, 4)
max_iter, alpha, dt, discount, step, scale, mode = 500, .5, 0.008, 1.0, 1, 100, 'proj'
Q = torch.diag(torch.Tensor([.05, 5, .1, .1])).repeat(sim_params.nsim, 1, 1).to(device)
R = torch.diag(torch.Tensor([0.0001])).repeat(sim_params.nsim, 1, 1).to(device)
Qf = torch.diag(torch.Tensor([5, 300, 10, 10])).repeat(sim_params.nsim, 1, 1).to(device)
lambdas = torch.ones((sim_params.ntime-0, sim_params.nsim, 1, 1))
cartpole = Cartpole(sim_params.nsim, cp_params, device)


def build_discounts(lambdas: torch.Tensor, discount: float):
    for i in range(lambdas.shape[0]):
        lambdas[i, :, :, :] *= (discount)**i

    return lambdas.clone()


def state_encoder(x: torch.Tensor):
    b, r, c = x.shape
    x = x.reshape((b, r*c))
    qc, qp, v = x[:, 0].clone().unsqueeze(1), x[:, 1].clone().unsqueeze(1), x[:, 2:].clone()
    qp = torch.cos(qp) - 1
    return torch.cat((qc, qp, v), 1).reshape((b, r, c))


def batch_state_encoder(x: torch.Tensor):
    t, b, r, c = x.shape
    x = x.reshape((t*b, r*c))
    qc, qp, v = x[:, 0].clone().unsqueeze(1), x[:, 1].clone().unsqueeze(1), x[:, 2:].clone()
    qp = torch.cos(qp) - 1
    return torch.cat((qc, qp, v), 1).reshape((t, b, r, c))


class NNValueFunction(nn.Module):
    def __init__(self, n_in):
        super(NNValueFunction, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(n_in+1, 256),
            nn.Softplus(beta=5),
            nn.Linear(256, 1)
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.nn.apply(init_weights)

    def forward(self, t, x):
        time = torch.ones((sim_params.nsim, 1, 1)) * t
        aug_x = torch.cat((x, time), dim=2)
        return self.nn(aug_x)


def loss_func(x: torch.Tensor):
    x = state_encoder(x)
    return x @ Q @ x.mT


nn_value_func = NNValueFunction(sim_params.nqv).to(device)


def backup_loss(x: torch.Tensor):
    t, nsim, r, c = x.shape
    x_final = x[-1].view(1, nsim, r, c).clone()
    x_init = x[0].view(1, nsim, r, c).clone()
    x_final_w = batch_state_encoder(x_final)
    x_init_w = batch_state_encoder(x_init)
    value_final = nn_value_func(0, x_final_w).squeeze()
    value_init = nn_value_func(0, x_init_w).squeeze()

    return -value_init + value_final


def batch_dynamics_loss(x, acc, alpha=1):
    t, b, r, c = x.shape
    x_reshape = x.reshape((t*b, 1, sim_params.nqv))
    a_reshape = acc.reshape((t*b, 1, sim_params.nv))
    acc_real = cartpole(x_reshape, a_reshape).reshape(x.shape)[:, :, :, sim_params.nv:]
    l_run = torch.sum((acc - acc_real)**2, 0).squeeze()
    return torch.mean(l_run) * alpha


def batch_state_loss(x: torch.Tensor):
    x = batch_state_encoder(x)
    t, nsim, r, c = x.shape
    x_run = x[:-1, :, :, :].view(t-1, nsim, r, c).clone()
    x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
    l_running = torch.sum(x_run @ Q @ x_run.mT, 0).squeeze()
    l_terminal = (x_final @ Qf @ x_final.mT).squeeze()

    return torch.mean(l_running + l_terminal)


def batch_inv_dynamics_loss(x, acc, alpha):
    x = x[:-1].clone()
    q, v = x[:, :, :, :sim_params.nq], x[:, :, :, sim_params.nq:]
    q = q.reshape((q.shape[0]*q.shape[1], 1, sim_params.nq))
    x_reshape = x.reshape((x.shape[0]*x.shape[1], 1, sim_params.nqv))
    M = cartpole._Mfull(q).reshape((x.shape[0], x.shape[1], sim_params.nv, sim_params.nv))
    C = cartpole._Cfull(x_reshape).reshape((x.shape[0], x.shape[1], sim_params.nv, sim_params.nv))
    Tg = cartpole._Tgrav(q).reshape((x.shape[0], x.shape[1], 1, sim_params.nq))
    Tf = cartpole._Tfric(v).reshape((x.shape[0], x.shape[1], 1, sim_params.nv))
    u_batch = (M @ acc.mT).mT + (C @ v.mT).mT - Tg + Tf
    return torch.sum(u_batch @ torch.linalg.inv(M) @ u_batch.mT, 0).squeeze() / scale


def loss_function(x, acc, alpha=1):
    l_ctrl, l_state, l_bellman = batch_inv_dynamics_loss(x, acc, alpha), batch_state_loss(x), backup_loss(x)
    loss = torch.mean(l_ctrl + l_state + l_bellman)
    return torch.maximum(loss, torch.zeros_like(loss))


dyn_system = ProjectedDynamicalSystem(
    nn_value_func, loss_func, sim_params, encoder=state_encoder, dynamics=cartpole, mode=mode, step=step, scale=scale, R=R
).to(device)
time = torch.linspace(0, (sim_params.ntime - 1) * dt, sim_params.ntime).to(device)
one_step = torch.linspace(0, dt, 2).to(device)
optimizer = torch.optim.AdamW(dyn_system.parameters(), lr=6e-3, amsgrad=True)
lambdas = build_discounts(lambdas, discount).to(device)

fig_3, p, r, width, height = init_fig_cp(0)

log = f"m-{mode}_d-{discount}_s-{step}"


def schedule_lr(optimizer, epoch, rate):
    pass
    # if epoch % rate == 0:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] *= (0.75 * (0.95 ** (epoch / rate)))

loss_buffer = []

if __name__ == "__main__":
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    qc_init = torch.FloatTensor(sim_params.nsim, 1, 1).uniform_(0, 0) * 2
    qp_init = torch.FloatTensor(sim_params.nsim, 1, 1).uniform_(torch.pi - 0.3, torch.pi + 0.3)
    qd_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nv).uniform_(-1, 1)
    x_init = torch.cat((qc_init, qp_init, qd_init), 2).to(device)
    iteration = 0
    alpha = 0

    while iteration < max_iter:
        optimizer.zero_grad()
        x_init = x_init[torch.randperm(sim_params.nsim)[:], :, :].clone()
        traj = odeint(dyn_system, x_init, time, method='euler', options=dict(step_size=dt))
        acc = compose_acc(traj[:, :, :, sim_params.nv:].clone(), dt)
        loss = loss_function(traj, acc, alpha)
        loss.backward()
        optimizer.step()
        schedule_lr(optimizer, iteration, 20)

        print(f"Epochs: {iteration}, Loss: {loss.item()}, lr: {get_lr(optimizer)}")

        selection = random.randint(0, sim_params.nsim - 1)
        fig_5 = plt.figure(5)
        ax_2 = plt.axes()
        ax_5 = plt.axes()
        ax_5.set_title('Loss')
        plt.plot(loss_buffer)
        fig_2 = plt.figure(2)
        plt.plot(acc[:, 0, 0, 0].cpu().detach())
        plt.pause(0.001)
        fig_2.clf()
        fig_5.clf()
        fig_1 = plt.figure(1)
        fig_1.clf()

        if iteration % 40 == 0 and iteration != 0:
            for i in range(sim_params.nsim):
                qpole = traj[:, i, 0, 1].cpu().detach()
                qdpole = traj[:, i, 0, 3].cpu().detach()
                plt.plot(qpole, qdpole)

            plt.pause(0.001)

            for i in range(0, sim_params.nsim, 60):
                selection = random.randint(0, sim_params.nsim - 1)
                cart = traj[:, selection, 0, 0].cpu().detach().numpy()
                pole = traj[:, selection, 0, 1].cpu().detach().numpy()
                animate_cartpole(cart, pole, fig_3, p, r, width, height, skip=3)


        iteration += 1

    model_scripted = torch.jit.script(dyn_system.value_func.clone().to('cpu'))  # Export to TorchScript
    model_scripted.save(f'{log}.pt')  # Save
    input()

