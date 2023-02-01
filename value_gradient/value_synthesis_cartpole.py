import math
import random
import mujoco
import torch
from models import Cartpole, ModelParams
from animations.cartpole import  init_fig_cp, animate_cartpole
from neural_value_synthesis_diffeq import *
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from utilities.mujoco_torch import SimulationParams

sim_params = SimulationParams(6, 4, 2, 2, 2, 1, 80, 240, 0.008)
cp_params = ModelParams(2, 2, 1, 4, 4)
prev_cost, diff, tol, max_iter, alpha, dt, n_bins, discount = 0, 100.0, 0, 2000, .5, 0.008, 3, 1.1
Q = torch.diag(torch.Tensor([5, 2, 0.01, 0.01])).repeat(sim_params.nsim, 1, 1).to(device)
R = torch.diag(torch.Tensor([0.0001])).repeat(sim_params.nsim, 1, 1).to(device)
Qf = torch.diag(torch.Tensor([5, 300, 10, 10])).repeat(sim_params.nsim, 1, 1).to(device)
lambdas = torch.ones((sim_params.ntime, sim_params.nsim, 1, 1))
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

def wrap_free_state(x: torch.Tensor):
    q, v = x[:, :, :sim_params.nq], x[:, :, sim_params.nq:]
    q_new = torch.cat((torch.sin(q[:, :, 1]), torch.cos(q[:, :, 1]) - 1, q[:, :, 0]), 1).unsqueeze(1)
    return torch.cat((q_new, v), 2)


def batch_wrap_free_state(x: torch.Tensor):
    q, v = x[:, :, :, :sim_params.nq], x[:, :, :, sim_params.nq:]
    q_new = torch.cat((torch.sin(q[:, :, :, 1]), torch.cos(q[:, :, :, 1]) - 1, q[:, :, :, 0]), 2).unsqueeze(2)
    return torch.cat((q_new, v), 3)


def bounded_state(x: torch.Tensor):
    qc, qp, qdc, qdp = x[:, :, 0].clone().unsqueeze(1), x[:, :, 1].clone().unsqueeze(1), x[:, :, 2].clone().unsqueeze(1), x[:, :, 3].clone().unsqueeze(1)
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
            nn.Linear(n_in, 32),
            nn.Softplus(beta=5),
            nn.Linear(32, 1),
        )

        def init_weights(net):
            if type(net) == nn.Linear:
                torch.nn.init.xavier_uniform(net.weight)

        self.nn.apply(init_weights)

    def forward(self, t, x):
        return self.nn(x)


def loss_func(x: torch.Tensor):
    x = state_encoder(x)
    return x @ Q @ x.mT


nn_value_func = NNValueFunction(sim_params.nqv).to(device)


def backup_loss(x: torch.Tensor):
    t, nsim, r, c = x.shape
    x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
    factor  = lambdas[-1, :, :, :].view(1, nsim, 1, 1).clone()
    x_final_w = batch_state_encoder(x_final)
    l_running = (x_final_w @ Q @ x_final_w.mT) * factor
    l_running = torch.sum(l_running, 0).squeeze()
    value = nn_value_func(0, x_final_w).squeeze()
    return torch.mean(torch.square(value - l_running))


def batch_dynamics_loss(x, acc, alpha=1):
    t, b, r, c = x.shape
    x_reshape = x.reshape((t*b, 1, sim_params.nqv))
    a_reshape = acc.reshape((t*b, 1, sim_params.nv))
    acc_real = cartpole(x_reshape, a_reshape).reshape(x.shape)[:, :, :, sim_params.nv:]
    l_run = torch.sum((acc - acc_real)**2, 0).squeeze()
    return torch.mean(l_run) * alpha


def batch_state_loss(x: torch.Tensor):
    x = batch_state_encoder(x)
    # t, nsim, r, c = x.shape
    # x_run = x[, :, :, :].view(t-1, nsim, r, c).clone()
    # x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
    l_running = (x @ Q @ x.mT) * lambdas
    l_running = torch.sum(l_running, 0).squeeze()
    # l_terminal = (x_final @ Qf @ x_final.mT).squeeze() * 0

    return torch.mean(l_running)

def batch_ctrl_loss(acc: torch.Tensor):
    qddc = acc[:, :, :, 0].unsqueeze(2).clone()
    l_ctrl = torch.sum(qddc @ R @ qddc.mT, 0).squeeze()
    return torch.mean(l_ctrl)


def batch_inv_dynamics_loss(x, acc, alpha):
    q, v = x[:, :, :, :sim_params.nq], x[:, :, :, sim_params.nq:]
    q = q.reshape((q.shape[0]*q.shape[1], 1, sim_params.nq))
    x_reshape = x.reshape((x.shape[0]*x.shape[1], 1, sim_params.nqv))
    M = cartpole._Mfull(q).reshape((x.shape[0], x.shape[1], sim_params.nv, sim_params.nv))
    C = cartpole._Cfull(x_reshape).reshape((x.shape[0], x.shape[1], sim_params.nv, sim_params.nv))
    Tg = cartpole._Tgrav(q).reshape((x.shape[0], x.shape[1], 1, sim_params.nq))
    u_batch = (M @ acc.mT).mT + (C @ v.mT).mT - Tg
    return torch.mean(torch.sum(u_batch @ torch.linalg.inv(M) @ u_batch.mT, 0).squeeze()) * 0.00001


def loss_function(x, acc, alpha=1):
    l_ctrl, l_state, l_bellman, l_dyn = batch_inv_dynamics_loss(x, acc, alpha), batch_state_loss(x), backup_loss(x), batch_dynamics_loss(x, acc, alpha)
    print(f"loss ctrl {l_ctrl}, loss state {l_state}, loss bellman {l_bellman}, loss dynamics {l_dyn}, alpha {alpha}")
    return l_ctrl + l_state + l_bellman + l_dyn


thetas = torch.linspace(torch.pi - 0.6, torch.pi + 0.6, n_bins)
mid_point = int(len(thetas)/2) + len(thetas) % 2
dyn_system = ProjectedDynamicalSystem(
    nn_value_func, loss_func, sim_params, encoder=state_encoder, dynamics=cartpole, mode='hjb'
).to(device)
time = torch.linspace(0, (sim_params.ntime - 1) * dt, sim_params.ntime).to(device)
optimizer = torch.optim.AdamW(dyn_system.parameters(), lr=.5e-2, amsgrad=True, weight_decay=0.03)
sc = torch.optim.lr_scheduler.ExponentialLR(optimizer, 1)
lambdas = build_discounts(lambdas, 1.01).to(device)
full_iteraiton = 0

fig_3, p, r, width, height = init_fig_cp(0)


if __name__ == "__main__":

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    for i in range(1, mid_point+1):
        nsim = sim_params.nsim
        qp_init1 = torch.FloatTensor(int(nsim/2), 1, 1).uniform_(thetas[0], thetas[i]) * 1
        qp_init2 = torch.FloatTensor(int(nsim/2), 1, 1).uniform_(thetas[-1-i], thetas[-1]) * 1
        qp_init = torch.cat((qp_init1, qp_init2), 0)
        qc_init = torch.FloatTensor(nsim, 1, 1).uniform_(0, 0) * 1
        qd_init = torch.FloatTensor(nsim, 1, sim_params.nv).uniform_(0, 0) * 1
        x_init = torch.cat((qc_init, qp_init, qd_init), 2).to(device)
        iteration = 0
        alpha = 0

        print(f"Theta range {thetas[0]} to {thetas[i]} and {thetas[-1-i]} to {thetas[-1]}")
        while iteration < max_iter:
            optimizer.zero_grad()
            traj = odeint(dyn_system, x_init, time, method='euler', options=dict(step_size=dt))
            acc = compose_acc(traj, dt)
            xxd = compose_xxd(traj, acc)
            loss = loss_function(traj, acc, alpha)
            loss.backward()
            optimizer.step()
            sc.step()
            alpha *= 1.03

            if alpha > 4:
                alpha = 4

            print(f"Epochs: {full_iteraiton}, Loss: {loss.item()}, iteration: {iteration % 10}, lr: {get_lr(optimizer)}")

            selection = random.randint(0, sim_params.nsim - 1)

            if iteration % 60 == 0 and iteration != 0:
                fig_1 = plt.figure(1)
                for i in range(sim_params.nsim):
                    qpole = traj[:, i, 0, 1].cpu().detach()
                    qdpole = traj[:, i, 0, 3].cpu().detach()
                    plt.plot(qpole, qdpole)

                plt.pause(0.001)
                fig_2 = plt.figure(2)
                ax_2 = plt.axes()
                plt.plot(traj[:, selection, 0, 0].cpu().detach())
                plt.plot(acc[:, selection, 0, 0].cpu().detach())
                plt.plot(acc[:, selection, 0, 1].cpu().detach())
                plt.pause(0.001)
                ax_2.set_title(loss.item())

                for i in range(0, sim_params.nsim, 10):
                    selection = random.randint(0, sim_params.nsim - 1)
                    cart = traj[:, selection, 0, 0].cpu().detach().numpy()
                    pole = traj[:, selection, 0, 1].cpu().detach().numpy()
                    animate_cartpole(cart, pole, fig_3, p, r, width, height, skip=2)

                fig_1.clf()
                fig_2.clf()

            if full_iteraiton == 360:
                model_scripted = torch.jit.script(dyn_system.value_func.to('cpu'))  # Export to TorchScript
                model_scripted.save('model_scripted.pt')  # Save
                input()

            iteration += 1
            full_iteraiton += 1