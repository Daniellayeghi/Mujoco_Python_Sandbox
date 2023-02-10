import random

import torch

from models import Cartpole, ModelParams
from animations.cartpole import init_fig_cp, animate_cartpole
from neural_value_synthesis_diffeq import *
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from utilities.mujoco_torch import SimulationParams

sim_params = SimulationParams(3, 2, 1, 1, 1, 1, 80, 501, 0.01)
prev_cost, diff, tol, max_iter, alpha, dt, n_bins, discount, step, scale, mode = 0, 100.0, 0, 250, .5, 0.01, 3, 1, 15, 100, 'hjb'
Q = torch.diag(torch.Tensor([1, .01])).repeat(sim_params.nsim, 1, 1).to(device)
R = torch.diag(torch.Tensor([1])).repeat(sim_params.nsim, 1, 1).to(device)
lambdas = torch.ones((sim_params.ntime, sim_params.nsim, 1, 1))


def build_discounts(lambdas: torch.Tensor, discount: float):
    for i in range(lambdas.shape[0]):
        lambdas[i, :, :, :] *= (discount)**i

    return lambdas.clone()


def state_encoder(x: torch.Tensor):
    return x


def batch_state_encoder(x: torch.Tensor):
    return x


def norm_cst(cst: torch.Tensor, dim=0):
    return cst


class NNValueFunction(nn.Module):
    def __init__(self, n_in):
        super(NNValueFunction, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(n_in, 32),
            nn.Softplus(beta=5),
            nn.Linear(32, 1),
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.nn.apply(init_weights)

    def forward(self, t, x):
        return self.nn(x)


nn_value_func = NNValueFunction(sim_params.nqv).to(device)


def loss_func(x: torch.Tensor, t):
    x = state_encoder(x)
    l = x @ Q @ x.mT
    l = (norm_cst(l) * (discount ** (t * sim_params.dt))).squeeze()
    return l


def state_loss(x: torch.Tensor):
    x = batch_state_encoder(x)
    # t, nsim, r, c = x.shape
    # x_run = x[, :, :, :].view(t-1, nsim, r, c).clone()
    # x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
    l_running = (x @ Q @ x.mT)
    l_running = torch.sum(l_running, 0) * lambdas
    return l_running


def state_loss_batch(x: torch.Tensor):
    l_running = state_loss(x)
    return torch.mean(l_running)


def inv_dynamics_reg(x: torch.Tensor, acc: torch.Tensor, alpha):
    u_batch = torch.linalg.inv(R) @ acc
    loss = u_batch @ R @ u_batch.mT
    return torch.sum(loss, 0)


def inv_dynamics_reg_batch(x: torch.Tensor, acc: torch.Tensor, alpha):
    l_ctrl = inv_dynamics_reg(x, acc, alpha)
    return torch.mean(l_ctrl) * 1/scale


def backup_loss(x: torch.Tensor, acc, alpha):
    t, nsim, r, c = x.shape
    x_initial =  batch_state_encoder(x[0, :, :, :].view(1, nsim, r, c).clone())
    x_final =  batch_state_encoder(x[-1, :, :, :].view(1, nsim, r, c).clone())
    x = batch_state_encoder(x[:-1].view(t-1, nsim, r, c).clone())
    acc = acc[:-1].view(t-1, nsim, r, sim_params.nu).clone()
    l_running = state_loss_batch(x) + inv_dynamics_reg_batch(x, acc, alpha)
    value_final = nn_value_func(0, x_final.squeeze()).squeeze()
    value_initial = nn_value_func(0, x_initial.squeeze()).squeeze()
    loss = torch.square(value_final - value_initial + l_running)
    return torch.mean(loss)


def lyapounov_goal_loss(x: torch.Tensor):
    t, nsim, r, c = x.shape
    x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
    x_final_w = batch_state_encoder(x_final)
    value = nn_value_func(0, x_final_w.squeeze())
    return torch.mean(value)


def loss_function_bellman(x, acc, alpha=1):
    l_bellman = backup_loss(x, acc, alpha)
    print(f"loss bellman {l_bellman}, alpha {alpha}")
    return l_bellman


def loss_function_lyapounov(x, acc, alpha=1):
    l_ctrl, l_state, l_lyap = inv_dynamics_reg_batch(x, acc, alpha), state_loss_batch(x), lyapounov_goal_loss(x)
    print(f"loss ctrl {l_ctrl}, loss state {l_state}, loss bellman {l_lyap}, alpha {alpha}")
    return l_ctrl + l_state + l_lyap


pos_arr = torch.linspace(-10, 10, 100).to(device)
vel_arr = torch.linspace(-10, 10, 100).to(device)
f_mat = torch.zeros((100, 100)).to(device)
[X, Y] = torch.meshgrid(pos_arr.squeeze().cpu(), vel_arr.squeeze().cpu())
time = torch.linspace(0, (sim_params.ntime - 1) * dt, sim_params.ntime).to(device)

dyn_system = ProjectedDynamicalSystem(
    nn_value_func, loss_func, sim_params, encoder=state_encoder, mode=mode, step=step, scale=scale
).to(device)


optimizer = torch.optim.AdamW(dyn_system.parameters(), lr=3e-2, amsgrad=True)
lambdas = build_discounts(lambdas, discount).to(device)

fig_3, p, r, width, height = init_fig_cp(0)

log = f"m-{mode}_d-{discount}_s-{step}"


def schedule_lr(optimizer, epoch, rate):
    if epoch % rate == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= (0.75 * (0.95 ** (epoch/rate)))


if __name__ == "__main__":

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    q_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nq).uniform_(-1, 1) * 10
    qd_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nq).uniform_(-1, 1) * 10
    x_init = torch.cat((q_init, qd_init), 2).to(device)
    trajectory = x_init.detach().clone().unsqueeze(0)
    iteration = 0

    while iteration < max_iter:
        optimizer.zero_grad()

        traj = odeint(
            dyn_system, x_init, time, method='euler',
            options=dict(step_size=dt), adjoint_atol=1e-9, adjoint_rtol=1e-9
        )

        acc = compose_acc(traj, dt)
        xxd = compose_xxd(traj, acc)
        traj = traj.clone()
        loss = loss_function_bellman(traj, acc, alpha)
        loss.backward()
        optimizer.step()

        print(f"Epochs: {iteration}, Loss: {loss.item()}, iteration: {iteration % 10}, lr: {get_lr(optimizer)}")

        if iteration % 20 == 1:
            with torch.no_grad():
                plot_2d_funcition(pos_arr, vel_arr, [X, Y], f_mat, nn_value_func, trace=traj, contour=True)

        iteration += 1

    model_scripted = torch.jit.script(dyn_system.value_func.clone().to('cpu'))  # Export to TorchScript
    model_scripted.save(f'{log}.pt')  # Save
