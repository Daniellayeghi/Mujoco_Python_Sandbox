import random
import numpy as np
import torch

from models import TwoLink, ModelParams
from neural_value_synthesis_diffeq import *
import matplotlib.pyplot as plt
from utilities.mujoco_torch import SimulationParams
from PSDNets import ICNN
from torchdiffeq_ctrl import odeint_adjoint as ctrl_odeint
from mj_renderer import *

sim_params = SimulationParams(6, 4, 2, 2, 2, 1, 10, 10, 0.01)
tl_params = ModelParams(2, 2, 1, 4, 4)
max_iter, alpha, dt, discount, step, scale, mode = 500, .5, 0.01, 1.0, 15, 10, 'inv'
Q = torch.diag(torch.Tensor([5, 5, .0001, .0001])).repeat(sim_params.nsim, 1, 1).to(device)
Qf = torch.diag(torch.Tensor([1000, 1000, 10, 10])).repeat(sim_params.nsim, 1, 1).to(device)
lambdas = torch.ones((sim_params.ntime-0, sim_params.nsim, 1, 1))
tl = TwoLink(sim_params.nsim, tl_params, device)
renderer = MjRenderer("../xmls/reacher.xml", dt=0.000001)

def build_discounts(lambdas: torch.Tensor, discount: float):
    for i in range(lambdas.shape[0]):
        lambdas[i, :, :, :] *= (discount)**i

    return lambdas.clone()


def state_encoder(x: torch.Tensor):
    b, r, c = x.shape
    x = x.reshape((b, r*c))
    q, v = x[:, :sim_params.nq].clone().unsqueeze(1), x[:, sim_params.nq:].clone().unsqueeze(1)
    q = torch.cos(q) - 1
    return torch.cat((q, v), 1).reshape((b, r, c))


def batch_state_encoder(x: torch.Tensor):
    t, b, r, c = x.shape
    x = x.reshape((t*b, r*c))
    q, v = x[:, :sim_params.nq].clone().unsqueeze(1), x[:, sim_params.nq:].clone().unsqueeze(1)
    q = torch.cos(q) - 1
    return torch.cat((q, v), 1).reshape((t, b, r, c))


class NNValueFunction(nn.Module):
    def __init__(self, n_in):
        super(NNValueFunction, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(n_in+1, 128),
            nn.Softplus(beta=5),
            nn.Linear(128, 1)
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.nn.apply(init_weights)

    def forward(self, t, x):
        time = torch.ones((sim_params.nsim, 1, 1)).to(device) * t
        aug_x = torch.cat((x, time), dim=2)
        return self.nn(aug_x)


def loss_quadratic(x, gain):
    return x @ gain @ x.mT


def loss_func(x: torch.Tensor):
    x = state_encoder(x)
    return loss_quadratic(x, Q)


nn_value_func = NNValueFunction(sim_params.nqv).to(device)


def backup_loss(x: torch.Tensor):
    t, nsim, r, c = x.shape
    x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
    x_init = x[0, :, :, :].view(1, nsim, r, c).clone()
    x_final_w = batch_state_encoder(x_final).reshape(nsim, r, c)
    x_init_w = batch_state_encoder(x_init).reshape(nsim, r, c)
    value_final = nn_value_func(0, x_final_w).squeeze()
    value_init = nn_value_func((sim_params.ntime - 1) * dt, x_init_w).squeeze()

    return -value_init + value_final


def batch_state_loss(x: torch.Tensor):
    x = batch_state_encoder(x)
    t, nsim, r, c = x.shape
    x_run = x[:-1, :, :, :].view(t-1, nsim, r, c).clone()
    x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
    l_running = loss_quadratic(x_run, Q).squeeze()
    l_terminal = loss_quadratic(x_final, Qf).squeeze()

    return torch.cat((l_running, l_terminal.unsqueeze(0)), dim=0)


def batch_inv_dynamics_loss(x, acc, alpha):
    q, v = x[:, :, :, :sim_params.nq], x[:, :, :, sim_params.nq:]
    q = q.reshape((q.shape[0]*q.shape[1], 1, sim_params.nq))
    x_reshape = x.reshape((x.shape[0]*x.shape[1], 1, sim_params.nqv))
    M = tl._Mfull(q).reshape((x.shape[0], x.shape[1], sim_params.nv, sim_params.nv))
    C = tl._Tbias(x_reshape).reshape((x.shape[0], x.shape[1], 1, sim_params.nv))
    u_batch = (M @ acc.mT).mT - C
    return u_batch @ torch.linalg.inv(M) @ u_batch.mT / scale


def loss_function(x, acc, alpha=1):
    l_run = torch.sum(batch_inv_dynamics_loss(x, acc, alpha) + batch_state_loss(x), dim=0)
    l_bellman = backup_loss(x)
    return torch.mean(torch.square(l_run + l_bellman))


dyn_system = ProjectedDynamicalSystem(
    nn_value_func, loss_func, sim_params, encoder=state_encoder, dynamics=tl, mode=mode, step=step, scale=scale
).to(device)
time = torch.linspace(0, (sim_params.ntime - 1) * dt, sim_params.ntime).to(device).requires_grad_(True)
one_step = torch.linspace(0, dt, 2).to(device)
optimizer = torch.optim.AdamW(dyn_system.parameters(), lr=2e-3, amsgrad=True)
lambdas = build_discounts(lambdas, discount).to(device)


log = f"m-{mode}_d-{discount}_s-{step}"


def schedule_lr(optimizer, epoch, rate):
    pass
    if epoch == 250:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.25

loss_buffer = []

if __name__ == "__main__":

    def transform_coordinates_tl(traj: torch.Tensor):
       traj[:, :, :, 1] = torch.pi - (traj[:, :, :, 0] + (torch.pi - traj[:, :, :, 1]))
       return traj

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    q_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nq).uniform_(-torch.pi, torch.pi)
    qd_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nv).uniform_(-1, 1) * 0
    x_init = torch.cat((q_init, qd_init), 2).to(device)
    iteration = 0
    alpha = 0

    try:
        while iteration < max_iter:
            optimizer.zero_grad()
            x_init = x_init[torch.randperm(sim_params.nsim)[:], :, :].clone()
            traj, dtrj_dt = ctrl_odeint(dyn_system, x_init, time, method='euler', options=dict(step_size=dt))
            acc = dtrj_dt[:, :, :, sim_params.nv:].clone()
            loss = loss_function(traj, acc, alpha)
            loss_buffer.append(loss.item())
            loss.backward()
            optimizer.step()
            schedule_lr(optimizer, iteration, 20)

            print(f"Epochs: {iteration}, Loss: {loss.item()}, lr: {get_lr(optimizer)}")

            ax3 = plt.subplot(212)
            ax3.clear()
            ax3.margins()  # Values in (-0.5, 0.0) zooms in to center
            ax3.plot(loss_buffer)
            ax3.set_title('Loss')

            if iteration % 50 == 0:
                ax1 = plt.subplot(222)
                ax2 = plt.subplot(221)
                ax1.clear()
                ax2.clear()
                for i in range(0, sim_params.nsim, 20):
                    selection = random.randint(0, sim_params.nsim - 1)
                    ax1.margins(0.05)  # Default margin is 0.05, value 0 means fit
                    ax1.plot(traj[:, selection, 0, 0].cpu().detach())
                    ax1.plot(traj[:, selection, 0, 1].cpu().detach())
                    ax1.set_title('Pos')
                    ax2 = plt.subplot(221)
                    ax2.margins(0.05)  # Values >0.0 zoom out
                    ax2.plot(dtrj_dt[:, selection, 0, 0].cpu().detach())
                    ax2.plot(dtrj_dt[:, selection, 0, 1].cpu().detach())
                    ax2.set_title('Acc')
                    traj_tl_mj = transform_coordinates_tl(traj.clone())
                    renderer.render(traj_tl_mj[:, selection, 0, :tl_params.nq].cpu().detach().numpy())

            iteration += 1
            plt.pause(0.001)


        model_scripted = torch.jit.script(dyn_system.value_func.to('cpu'))  # Export to TorchScript
        model_scripted.save(f'{log}.pt')  # Save
        np.save(f'{log}_loss', np.array(loss_buffer))# Save

        input()
    except KeyboardInterrupt:
        print("########## Saving Trace ##########")
        model_scripted = torch.jit.script(dyn_system.value_func.to('cpu'))  # Export to TorchScript
        model_scripted.save(f'{log}_except.pt')
        np.save(f'{log}_loss_except', np.array(loss_buffer))# Save
