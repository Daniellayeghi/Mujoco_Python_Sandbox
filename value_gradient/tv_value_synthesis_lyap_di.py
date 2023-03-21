import random
import numpy as np
import torch

from models import DoubleIntegrator, ModelParams
from mj_renderer import *
from neural_value_synthesis_diffeq import *
import matplotlib.pyplot as plt
from torchdiffeq_ctrl import odeint_adjoint as odeint
from utilities.mujoco_torch import SimulationParams
from PSDNets import ReHU, MakePSD, ICNN, PosDefICNN
from time_search import optimal_time
import wandb

di_params = ModelParams(1, 1, 1, 2, 2)
sim_params = SimulationParams(3, 2, 1, 1, 1, 1, 50, 135, 0.01)
di = DoubleIntegrator(sim_params.nsim, di_params, device)
max_iter, max_time, alpha, dt, discount, step, scale, mode = 500, 501, .5, 0.01, 1, 1, 2, 'proj'
Q = torch.diag(torch.Tensor([1, .01])).repeat(sim_params.nsim, 1, 1).to(device) * 10
Qf = torch.diag(torch.Tensor([1, .01])).repeat(sim_params.nsim, 1, 1).to(device) * 1000
R = torch.diag(torch.Tensor([.5])).repeat(sim_params.nsim, 1, 1).to(device)
lambdas = torch.ones((sim_params.ntime, sim_params.nsim, 1, 1))
renderer = MjRenderer("../xmls/pointmass.xml")

wandb.init(project='di_lyap', entity='lonephd')


def plot_2d_funcition(xs: torch.Tensor, ys: torch.Tensor, xy_grid, f_mat, func, trace=None, contour=True):
    assert len(xs) == len(ys)
    trace = trace.detach().clone().cpu().squeeze()
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            in_tensor = torch.tensor((x, y)).view(1, 1, 2).float().to(device)
            f_mat[i, j] = func(0, in_tensor).detach().squeeze()

    [X, Y] = xy_grid
    f_mat = f_mat.cpu()
    plt.clf()
    ax = plt.axes()
    if contour:
        ax.contourf(X, Y, f_mat, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    else:
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, f_mat, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('surface')
    ax.set_xlabel('Pos')
    ax.set_ylabel('Vel')
    n_plots = trace.shape[1]
    for i in range(n_plots):
        ax.plot(trace[:, i, 0], trace[:, i, 1])
    plt.pause(0.001)


def build_discounts(lambdas: torch.Tensor, discount: float):
    for i in range(lambdas.shape[0]):
        lambdas[i, :, :, :] *= (discount)**i

    return lambdas.clone()


def state_encoder(x: torch.Tensor):
    return x


def batch_state_encoder(x: torch.Tensor):
    return x


class NNValueFunction(nn.Module):
    def __init__(self, n_in):
        super(NNValueFunction, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.Softplus(beta=5),
            nn.Linear(64, 1),
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.nn.apply(init_weights)

    def forward(self, t, x):
        nsim, c = x.shape[0], x.shape[2]
        time = torch.ones((nsim, 1, 1)).to(device) * t
        aug_x = torch.cat((x, time), dim=2).reshape(nsim, c+1)
        return self.nn(aug_x)


def loss_func(x: torch.Tensor):
    x = state_encoder(x)
    return x @ Q @ x.mT

import torch.nn.functional as F
# nn_value_func = PosDefICNN([sim_params.nqv+1, 64, 64, 1]).to(device)
nn_value_func = NNValueFunction(sim_params.nqv+1).to(device)
# nn_value_func = MakePSD(ICNN([sim_params.nqv+1, 64, 64, 1], ReHU(0.01)), sim_params.nqv+1, eps=0.005, d=1)

def batch_state_loss(x: torch.Tensor):
    x = batch_state_encoder(x)
    t, nsim, r, c = x.shape
    x_run = x[:-1].view(t-1, nsim, r, c).clone()
    x_final = x[-1].view(1, nsim, r, c).clone()
    l_running = (x_run @ Q @ x_run.mT)
    l_terminal = (x_final @ Qf @ x_final.mT)
    l_running = torch.sum(l_running, 0) * lambdas
    return l_running + l_terminal


def inv_dynamics_reg(acc: torch.Tensor, alpha):
    u_batch = acc
    loss = u_batch @ R @ u_batch.mT
    return torch.sum(loss, 0)


def backup_loss(x: torch.Tensor):
    t, nsim, r, c = x.shape
    x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
    x_init = x[0, :, :, :].view(1, nsim, r, c).clone()
    x_final_w = batch_state_encoder(x_final).reshape(nsim, r, c)
    x_init_w = batch_state_encoder(x_init).reshape(nsim, r, c)
    value_final = nn_value_func((sim_params.ntime - 1) * dt, x_final_w).squeeze()
    value_init = nn_value_func(0, x_init_w).squeeze()

    return -value_init + value_final

def value_terminal_loss(x: torch.Tensor):
    t, nsim, r, c = x.shape
    x_final = x[-1].view(1, nsim, r, c).clone().reshape(nsim, r, c)
    value_final = nn_value_func(0, x_final).squeeze()
    return value_final

def batch_inv_dynamics_loss(acc, alpha):
    acc = acc[:-1, :, :, sim_params.nv:].clone()
    l_ctrl = inv_dynamics_reg(acc, alpha)
    return torch.mean(l_ctrl) * 1/scale


def loss_function(x, acc, alpha=1):
    l_ctrl, l_state, l_bellman, l_terminal = batch_inv_dynamics_loss(acc, alpha), batch_state_loss(x), backup_loss(x), value_terminal_loss(x) * 1000
    loss = torch.mean(l_ctrl + l_state + l_bellman + l_terminal)
    return torch.maximum(loss, torch.zeros_like(loss))


pos_arr = torch.linspace(-5, 5, 100).to(device)
vel_arr = torch.linspace(-5, 5, 100).to(device)
f_mat = torch.zeros((100, 100)).to(device)
[X, Y] = torch.meshgrid(pos_arr.squeeze().cpu(), vel_arr.squeeze().cpu())

dyn_system = ProjectedDynamicalSystem(
    nn_value_func, loss_func, sim_params, encoder=state_encoder, dynamics=di, mode=mode, step=step, scale=scale, R=R
).to(device)

optimizer = torch.optim.AdamW(dyn_system.parameters(), lr=1e-2, amsgrad=True)
lambdas = build_discounts(lambdas, discount).to(device)


log = f"m-{mode}_d-{discount}_s-{step}"
wandb.watch(dyn_system, loss_function, log="all")


def schedule_lr(optimizer, epoch, rate):
    pass
    # if epoch % rate == 0:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] *= (0.75 * (0.95 ** (epoch/rate)))


if __name__ == "__main__":

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    q_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nq).uniform_(-1, 1) * 5.5
    qd_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nq).uniform_(-1, 1) * 2
    x_init = torch.cat((q_init, qd_init), 2).to(device)
    trajectory = x_init.detach().clone().unsqueeze(0)
    iteration = 0

    while iteration < max_iter:
        optimizer.zero_grad()
        time = torch.linspace(0, (sim_params.ntime - 1) * dt, sim_params.ntime).to(device)
        x_init = x_init[torch.randperm(sim_params.nsim)[:], :, :].clone()
        traj, dtraj_dt = odeint(dyn_system, x_init, time, method='euler', options=dict(step_size=dt))
        acc = dtraj_dt[:, :, :, sim_params.nv:]
        loss = loss_function(traj, dtraj_dt, alpha)
        loss.backward()
        sim_params.ntime, update = optimal_time(sim_params.ntime, max_time, dt, loss_function, x_init, dyn_system, loss)
        optimizer.step()
        schedule_lr(optimizer, iteration, 60)
        wandb.log({'epoch': iteration + 1, 'loss': loss.item()})

        print(f"Epochs: {iteration}, Loss: {loss.item()}, lr: {get_lr(optimizer)}, time: {sim_params.ntime} \n")

        if iteration % 5 == 1:
            plot_2d_funcition(pos_arr, vel_arr, [X, Y], f_mat, nn_value_func, trace=traj, contour=True)

        if iteration % 20 == 1:
            with torch.no_grad():
                plot_2d_funcition(pos_arr, vel_arr, [X, Y], f_mat, nn_value_func, trace=traj, contour=True)
                renderer.render(traj[:, 0, 0, :sim_params.nq].cpu().detach().numpy())

        iteration += 1

    model_scripted = torch.jit.script(dyn_system.value_func.clone().to('cpu'))  # Export to TorchScript
    model_scripted.save(f'{log}.pt')  # Save
