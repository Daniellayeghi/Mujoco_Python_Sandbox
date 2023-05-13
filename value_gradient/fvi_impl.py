import random
import numpy as np
import torch

from models import Cartpole, ModelParams
from animations.cartpole import init_fig_cp, animate_cartpole
from neural_value_synthesis_diffeq import *
import matplotlib.pyplot as plt
from torchdiffeq_ctrl import odeint_adjoint as odeint
from utilities.mujoco_torch import SimulationParams
from fitted_value_iteration import FVI
from time_search import optimal_time
from PSDNets import ICNN, MakePSD, PosDefICNN, ICNNvanilla

import wandb
from mj_renderer import *

wandb.init(project='cartpole_lyap', entity='lonephd')

sim_params = SimulationParams(6, 4, 2, 2, 2, 1, 100, 100, 0.01)
cp_params = ModelParams(2, 2, 1, 4, 4)
max_iter, max_time, dt, discount, step, scale, mode = 500, 200, 0.01, 20, .005, 10, 'fwd'
Q = torch.diag(torch.Tensor([0.5, 5, 0, 0])).repeat(sim_params.nsim, 1, 1).to(device)
R = torch.diag(torch.Tensor([0.0001])).repeat(sim_params.nsim, 1, 1).to(device)
Qf = torch.diag(torch.Tensor([1000, 500, 10, 50])).repeat(sim_params.nsim, 1, 1).to(device)
cartpole = Cartpole(sim_params.nsim, cp_params, device)
renderer = MjRenderer("../xmls/cartpole.xml", 0.0001)


def build_discounts(lambdas: torch.Tensor, discount: float):
    for i in range(lambdas.shape[0]):
        lambdas[i, :, :, :] *= (discount)**i

    return lambdas.clone()


def atan2_encoder(x: torch.Tensor):
    return torch.pi ** 2 * torch.sin(x)


def state_encoder(x: torch.Tensor):
    b, r, c = x.shape
    x = x.reshape((b, r*c))
    qc, qp, v = x[:, 0].clone().unsqueeze(1), x[:, 1].clone().unsqueeze(1), x[:, 2:].clone()
    qp = atan2_encoder(qp)
    return torch.cat((qc, qp, v), 1).reshape((b, r, c))


def batch_state_encoder(x: torch.Tensor):
    t, b, r, c = x.shape
    x = x.reshape((t*b, r*c))
    qc, qp, v = x[:, 0].clone().unsqueeze(1), x[:, 1].clone().unsqueeze(1), x[:, 2:].clone()
    qp = atan2_encoder(qp)
    return torch.cat((qc, qp, v), 1).reshape((t, b, r, c))


class NNValueFunction(nn.Module):
    def __init__(self, n_in):
        super(NNValueFunction, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.Softplus(beta=5),
            nn.Linear(128, 1)
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.uniform_(m.bias)

        self.nn.apply(init_weights)

    def forward(self, t, x):
        x = x.reshape(x.shape[0], sim_params.nqv)
        b = x.shape[0]
        return self.nn(x).reshape(b, 1, 1)


nn_value_func = NNValueFunction(sim_params.nqv).to(device)


def loss_func(x: torch.Tensor):
    x = state_encoder(x)
    return x @ Q @ x.mT


def loss_quadratic(x, gain):
    return x @ gain @ x.mT


def batch_state_cst(x: torch.Tensor):
    x = batch_state_encoder(x)
    t, nsim, r, c = x.shape
    x_run = x[:-4, :, :, :].view(t-4, nsim, r, c).clone()
    x_final = x[-4:-1, :, :, :].view(3, nsim, r, c).clone()
    l_running = (loss_quadratic(x_run, Q)).squeeze()
    l_terminal = loss_quadratic(x_final, Qf).squeeze()

    return torch.cat((l_running, l_terminal), dim=0)


def batch_inv_dynamics_cst(x, acc, alpha):
    x, acc = x[:-1].clone(), acc[:-1, :, :, sim_params.nv:].clone()
    q, v = x[:, :, :, :sim_params.nq], x[:, :, :, sim_params.nq:]
    q = q.reshape((q.shape[0]*q.shape[1], 1, sim_params.nq))
    x_reshape = x.reshape((x.shape[0]*x.shape[1], 1, sim_params.nqv))
    M = cartpole._Mfull(q).reshape((x.shape[0], x.shape[1], sim_params.nv, sim_params.nv))
    C = cartpole._Tbias(x_reshape).reshape((x.shape[0], x.shape[1], 1, sim_params.nv))
    Tf = cartpole._Tfric(v).reshape((x.shape[0], x.shape[1], 1, sim_params.nv))
    u_batch = ((M @ acc.mT).mT - C + Tf) * torch.Tensor([1, 0]).to(device)
    return u_batch @ torch.linalg.inv(M) @ u_batch.mT / scale

def value_expansion(x):
    value = torch.zeros((x.shape[0], x.shape[1], 1))
    for t in range(x.shape[0]):
        value[t] = torch.sum(x[t:], dim=0)

    return value

def value_terminal_loss(x: torch.Tensor):
    t, nsim, r, c = x.shape
    x_final = x[-1].view(1, nsim, r, c).clone()
    x_final_w = batch_state_encoder(x_final).reshape(nsim, r, c)
    value_final = nn_value_func((t - 1) * dt, x_final_w).squeeze()
    return value_final


def loss(x, acc, alpha=1):
    t_loss = value_terminal_loss(x)
    u_loss = batch_inv_dynamics_cst(x, acc, alpha).squeeze()
    u_loss = torch.cat((u_loss, torch.zeros_like(u_loss[-1]).unsqueeze(0)), dim=0)
    x_loss = torch.cat((batch_state_cst(x) , t_loss.unsqueeze(0)), dim=0)
    loss = x_loss + u_loss
    return loss

def value_targets(loss):
    values = torch.zeros_like(loss)
    for i in range(loss.shape[1]):
        values[:, i] = torch.sum(loss[:, i:], dim=1)
    return values

dyn_system = ProjectedDynamicalSystem(
    nn_value_func, loss_func, sim_params, encoder=state_encoder, dynamics=cartpole, mode=mode, step=step, scale=scale, R=R
).to(device)

time = torch.linspace(0, (sim_params.ntime - 1) * dt, sim_params.ntime).to(device)
one_step = torch.linspace(0, dt, 2).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.AdamW(nn_value_func.parameters(), lr=1e-4, amsgrad=True)
fvi = FVI(nn_value_func, loss_function, optimizer, 100, 2)

fig_3, p, r, width, height = init_fig_cp(0)

log = f"m-{mode}_d-{discount}_s-{step}"


if __name__ == "__main__":

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    qc_init = torch.FloatTensor(sim_params.nsim, 1, 1).uniform_(0, 0)
    qp_init = torch.FloatTensor(sim_params.nsim, 1, 1).uniform_(torch.pi - 0.3, torch.pi + 0.3)
    qd_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nv).uniform_(-1, 1)
    x_init = torch.cat((qc_init, qp_init, qd_init), 2).to(device)
    iteration = 0
    alpha = 0

    while iteration < max_iter:
        with torch.no_grad():
            x_init = x_init[torch.randperm(sim_params.nsim)[:], :, :].clone()
            traj, dtrj_dt = odeint(dyn_system, x_init, time, method='euler', options=dict(step_size=dt))
            acc = dtrj_dt[:, :, :, sim_params.nv:]
            l = loss(traj, dtrj_dt, alpha)
            vs = value_targets(l)
        with torch.set_grad_enabled(True):
            fvi.train(batch_state_encoder(traj), vs, cat=False)

        # with torch.no_grad():
        #     next, _ = odeint(
        #         dyn_system, x_init, one_step, method='euler', options=dict(step_size=dt), adjoint_atol=1e-9, adjoint_rtol=1e-9
        #     )
        #     x_init = next[-1].detach().clone().reshape(sim_params.nsim, 1, sim_params.nqv)

        if iteration % 1 == 0:
            ax1 = plt.subplot(122)
            ax2 = plt.subplot(121)
            ax1.clear()
            ax2.clear()
            for i in range(0, sim_params.nsim, 100):
                selection = random.randint(0, sim_params.nsim - 1)
                ax1.margins(0.05)  # Default margin is 0.05, value 0 means fit
                ax1.plot(traj[:, selection, 0, 1].cpu().detach())
                ax1.set_title('Pos')
                ax2 = plt.subplot(221)
                ax2.margins(0.05)  # Values >0.0 zoom out
                ax2.plot(acc[:, selection, 0, 0].cpu().detach())
                ax2.plot(acc[:, selection, 0, 1].cpu().detach())
                ax2.set_title('Acc')
                renderer.render(traj[:, selection, 0, :sim_params.nq].cpu().detach().numpy())

        iteration += 1
        plt.pause(0.001)

    model_scripted = torch.jit.script(fvi._net.clone().to('cpu'))  # Export to TorchScript
    model_scripted.save(f'{log}.pt')  # Save
    input()

