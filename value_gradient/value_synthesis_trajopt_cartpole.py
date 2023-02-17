import random
import numpy as np
from models import Cartpole, ModelParams
from animations.cartpole import init_fig_cp, animate_cartpole
from neural_value_synthesis_diffeq import *
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from utilities.mujoco_torch import SimulationParams
from PSDNets import PosDefICNN
sim_params = SimulationParams(6, 4, 2, 2, 2, 1, 200, 240, 0.008)
cp_params = ModelParams(2, 2, 1, 4, 4)
prev_cost, diff, tol, max_iter, alpha, dt, n_bins, discount, step, scale, mode = 0, 100.0, 0, 500, .5, 0.008, 3, 1.0, 15, 10000, 'fwd'
Q = torch.diag(torch.Tensor([.05, 5, .1, .1])).repeat(sim_params.nsim, 1, 1).to(device)
Qf = torch.diag(torch.Tensor([5, 300, 10, 10])).repeat(sim_params.nsim, 1, 1).to(device)
R = torch.diag(torch.Tensor([1])).repeat(sim_params.nsim, 1, 1).to(device)
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
            nn.Linear(n_in, 64),
            nn.Softplus(),
            nn.Linear(64, 1),
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.nn.apply(init_weights)

    def forward(self, t, x):
        return self.nn(x)


nn_value_func = NNValueFunction(sim_params.nqv).to(device)
# nn_value_func = PosDefICNN([sim_params.nqv, 128, 128, 1], eps=0.01, negative_slope=0.3)

def loss_func(x: torch.Tensor, t):
    x = state_encoder(x)
    l = x @ Q @ x.mT
    return l


def state_loss(x: torch.Tensor):
    x = batch_state_encoder(x)
    l_running = (x @ Q @ x.mT)
    l_running = torch.sum(l_running, 0) * lambdas
    return l_running


def state_loss_batch(x: torch.Tensor):
    x = batch_state_encoder(x)
    t, nsim, r, c = x.shape
    x_run = x[:, :, :, :].view(t, nsim, r, c).clone()
    x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
    l_terminal = x_final @ Qf @ x_final.mT
    l_running = (x_run @ Q @ x_run.mT) #* lambdas
    l_running = torch.sum(l_running, 0).squeeze()
    return torch.mean(l_running + l_terminal)


def inv_dynamics_reg(x: torch.Tensor, acc: torch.Tensor, alpha):
    q, v = x[:, :, :, :sim_params.nq], x[:, :, :, sim_params.nq:]
    q = q.reshape((q.shape[0] * q.shape[1], 1, sim_params.nq))
    x_reshape = x.reshape((x.shape[0] * x.shape[1], 1, sim_params.nqv))
    M = cartpole._Mfull(q).reshape((x.shape[0], x.shape[1], sim_params.nv, sim_params.nv))
    C = cartpole._Cfull(x_reshape).reshape((x.shape[0], x.shape[1], sim_params.nv, sim_params.nv))
    Tg = cartpole._Tgrav(q).reshape((x.shape[0], x.shape[1], 1, sim_params.nq))
    u_batch = (M @ acc.mT).mT + (C @ v.mT).mT - Tg
    loss = u_batch @ torch.linalg.inv(M) @ u_batch.mT
    return torch.sum(loss, 0)


def ctrl_reg(x: torch.Tensor, acc: torch.Tensor, alpha):
    loss = acc @ R @ acc.mT
    return torch.sum(loss, 0)

def inv_dynamics_reg_batch(x: torch.Tensor, acc: torch.Tensor, alpha):
    l_ctrl = inv_dynamics_reg(x, acc, alpha)
    return torch.mean(l_ctrl) * 1/scale


def ctrl_reg_batch(x: torch.Tensor, acc: torch.Tensor, alpha):
    l_ctrl = ctrl_reg(x, acc, alpha)
    return torch.mean(l_ctrl) * 1/scale


def backup_loss(x: torch.Tensor, acc, alpha):
    t, nsim, r, c = x.shape
    x_initial =  batch_state_encoder(x[0, :, :, :].view(1, nsim, r, c).clone())
    x_final =  batch_state_encoder(x[-1, :, :, :].view(1, nsim, r, c).clone())
    x_run = batch_state_encoder(x[:-1].view(t-1, nsim, r, c).clone())
    acc = acc[:-1].view(t-1, nsim, r, 1).clone()
    l_running = state_loss_batch(x) + ctrl_reg_batch(x_run, acc, alpha)
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


def batch_dynamics_loss(x, acc, alpha=1):
    t, b, r, c = x.shape
    x_reshape = x.reshape((t*b, 1, sim_params.nqv))
    a_reshape = acc.reshape((t*b, 1, sim_params.nv))
    acc_real = cartpole(x_reshape, a_reshape).reshape(x.shape)[:, :, :, sim_params.nv:]
    l_run = (acc - acc_real)**2
    l_run = torch.sum(l_run, 0)
    return torch.mean(l_run) * alpha * 0


def batch_ctrl_loss(acc: torch.Tensor):
    qddc = acc[:, :, :, 0].unsqueeze(2).clone()
    l_ctrl = qddc @ R @ qddc.mT
    l_ctrl = torch.sum(l_ctrl, 0)
    return torch.mean(l_ctrl)


def loss_function_bellman(x, acc, alpha=1):
    l_bellman = backup_loss(x, acc, alpha)
    print(f"loss bellman {l_bellman} alpha {alpha}")
    return l_bellman


def loss_function_lyapounov(x, acc, alpha=1):
    l_ctrl, l_state, l_lyap = inv_dynamics_reg_batch(x, acc, alpha), state_loss_batch(x), lyapounov_goal_loss(x)
    print(f"loss ctrl {l_ctrl}, loss state {l_state}, loss bellman {l_lyap}, alpha {alpha}")
    return l_ctrl + l_state + l_lyap


dyn_system = ProjectedDynamicalSystem(
    nn_value_func, loss_func, sim_params, encoder=state_encoder, dynamics=cartpole, mode=mode, step=step, scale=scale, R=R
).to(device)
time = torch.linspace(0, (sim_params.ntime - 1) * dt, sim_params.ntime).to(device)
one_step = torch.linspace(0, dt, 2).to(device)
optimizer = torch.optim.AdamW(dyn_system.parameters(), lr=3e-4, amsgrad=True)
lambdas = build_discounts(lambdas, discount).to(device)

fig_3, p, r, width, height = init_fig_cp(0)

log = f"m-{mode}_d-{discount}_s-{step}"


def schedule_lr(optimizer, epoch, rate):
    pass
    if epoch % rate == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= (0.75 * (0.95 ** (epoch / rate)))

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
        dyn_system.collect = True
        traj = odeint(
            dyn_system, x_init, time, method='euler',
            options=dict(step_size=dt), adjoint_atol=1e-9, adjoint_rtol=1e-9
        )
        # acc = compose_acc(traj, dt)
        acc = dyn_system._acc_buffer.clone()
        loss = loss_function_bellman(traj, acc, alpha)
        loss_buffer.append(loss.item())
        dyn_system.collect = False
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

        if iteration % 10 == 0 and iteration != 0:
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

