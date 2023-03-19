import random
import torch
from models import DoubleCartpole, ModelParams, init_fig_dcp, animate_double_cartpole
from neural_value_synthesis_diffeq import *
import matplotlib.pyplot as plt
from torchdiffeq_ctrl import odeint_adjoint as odeint
from time_search import optimal_time
from utilities.mujoco_torch import SimulationParams
import wandb
from mj_renderer import MjRenderer
wandb.init(project='double_cartpole', entity='lonephd')
ren = MjRenderer('../xmls/double_cart_pole.xml')

sim_params = SimulationParams(9,6,3,3,1,1,1, 200, 0.01)
dcp_params = ModelParams(3, 3, 1, 6, 6)
max_iter, max_time, alpha, dt, n_bins, discount, step, scale, mode = 500, 400, .5, 0.01, 3, 1.025, 15, 50, 'inv'
Q = torch.diag(torch.Tensor([.5, .2, .2, 0, 0, 0])).repeat(sim_params.nsim, 1, 1).to(device)
R = torch.diag(torch.Tensor([0.0001])).repeat(sim_params.nsim, 1, 1).to(device)
Qf = torch.diag(torch.Tensor([1000, 500, 500, 10, 10, 10])).repeat(sim_params.nsim, 1, 1).to(device)
lambdas = torch.ones((sim_params.ntime, sim_params.nsim, 1, 1))
cartpole = DoubleCartpole(sim_params.nsim, dcp_params, device)


def atan2_encoder(x: torch.Tensor):
    return torch.atan2(torch.sin(x), torch.cos(x))


def build_discounts(lambdas: torch.Tensor, discount: float):
    for i in range(lambdas.shape[0]):
        lambdas[i, :, :, :] *= (discount)**i

    return lambdas.clone()


def state_encoder(x: torch.Tensor):
    b, r, c = x.shape
    x = x.reshape((b, r*c))
    qc, qp, v = x[:, 0].clone().unsqueeze(1), x[:, 1:sim_params.nq].clone(), x[:, sim_params.nq:].clone()
    qp = atan2_encoder(qp)
    return torch.cat((qc, qp, v), 1).reshape((b, r, c))


def batch_state_encoder(x: torch.Tensor):
    t, b, r, c = x.shape
    x = x.reshape((t*b, r*c))
    qc, qp, v = x[:, 0].clone().unsqueeze(1), x[:, 1:sim_params.nq].clone(), x[:, sim_params.nq:].clone()
    qp = atan2_encoder(qp)
    return torch.cat((qc, qp, v), 1).reshape((t, b, r, c))


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
            nn.Linear(n_in+1, 256),
            nn.Softplus(beta=5),
            nn.Linear(256, 1)
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.uniform_(m.bias)

        self.nn.apply(init_weights)

    def forward(self, t, x):
        time = torch.ones((sim_params.nsim, 1, 1)).to(device) * t
        aug_x = torch.cat((x, time), dim=2)
        return self.nn(aug_x)


nn_value_func = NNValueFunction(sim_params.nqv).to(device)


def loss_func(x: torch.Tensor, t):
    x = state_encoder(x)
    return x @ Q @ x.mT * (discount ** (t*sim_params.dt))


def loss_quadratic(x, gain):
    return x @ gain @ x.mT


def loss_func(x: torch.Tensor):
    x = state_encoder(x)
    return loss_quadratic(x, Q)


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
    x_run = x[:-4, :, :, :].view(t-4, nsim, r, c).clone()
    x_final = x[-4:-1, :, :, :].view(3, nsim, r, c).clone()
    l_running = (loss_quadratic(x_run, Q)).squeeze()
    l_terminal = loss_quadratic(x_final, Qf).squeeze()

    return torch.cat((l_running, l_terminal), dim=0)


def value_terminal_loss(x: torch.Tensor):
    t, nsim, r, c = x.shape
    x_final = x[-1].view(1, nsim, r, c).clone()
    x_final_w = batch_state_encoder(x_final).reshape(nsim, r, c)
    value_final = nn_value_func(0, x_final_w).squeeze()
    return value_final


def batch_ctrl_loss(acc: torch.Tensor):
    qddc = acc[:, :, :, 0].unsqueeze(2).clone()
    l_ctrl = torch.sum(qddc @ R @ qddc.mT, 0).squeeze()
    return torch.mean(l_ctrl)


def batch_inv_dynamics_loss(x, acc, alpha):
    x, acc = x[:-1].clone(), acc[:-1, :, :, sim_params.nv:].clone()
    q, v = x[:, :, :, :sim_params.nq], x[:, :, :, sim_params.nq:]
    q = q.reshape((q.shape[0]*q.shape[1], 1, sim_params.nq))
    x_reshape = x.reshape((x.shape[0]*x.shape[1], 1, sim_params.nqv))
    M = cartpole._Mfull(q).reshape((x.shape[0], x.shape[1], sim_params.nv, sim_params.nv))
    C = cartpole._Tbias(x_reshape).reshape((x.shape[0], x.shape[1], 1, sim_params.nv))
    u_batch = (M @ acc.mT).mT - C
    return u_batch @ torch.linalg.inv(M) @ u_batch.mT / scale


def loss_function(x, acc, alpha=1):
    l_run = torch.sum(batch_inv_dynamics_loss(x, acc, alpha) + batch_state_loss(x), dim=0)
    l_bellman = backup_loss(x)
    l_terminal = 1000 * value_terminal_loss(x) * 0
    return torch.mean(torch.square(l_run + l_bellman + l_terminal))


init_lr = 1e-2
dyn_system = ProjectedDynamicalSystem(
    nn_value_func, loss_func, sim_params, encoder=state_encoder, dynamics=cartpole, mode=mode, step=step, scale=scale
).to(device)
time = torch.linspace(0, (sim_params.ntime - 1) * dt, sim_params.ntime).requires_grad_(True).to(device)
optimizer = torch.optim.AdamW(dyn_system.parameters(), lr=init_lr, amsgrad=True)
lambdas = build_discounts(lambdas, discount).to(device)

fig_3, p, r, width, height = init_fig_dcp(0)

log = f"m-{mode}_d-{discount}_s-{step}"


def schedule_lr(optimizer, epoch, rate):
    pass
    lr = max(init_lr * (1.0 - epoch / 200) ** 2, 1e-3)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def transform_coordinates_dcp(traj: torch.Tensor):
    # traj[:, :, :, 1] = traj[:, :, :, 1] + 2 * (torch.pi - traj[:, :, :, 1])
    traj[:, :, :, 2] = torch.pi - (traj[:, :, :, 1] + (torch.pi - traj[:, :, :, 2]))
    return traj


if __name__ == "__main__":
    qp_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nq-1).uniform_(torch.pi, torch.pi) * 1
    qc_init = torch.FloatTensor(sim_params.nsim, 1, 1).uniform_(0, 0) * 1
    qd_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nv).uniform_(0, 0) * 1
    x_init = torch.cat((qc_init, qp_init, qd_init), 2).to(device)
    trajectory = x_init.detach().clone().unsqueeze(0)
    iteration = 0

    while iteration < max_iter:
        optimizer.zero_grad()
        traj, dtrajdt = odeint(dyn_system, x_init, time, method='euler', options=dict(step_size=dt))
        acc = dtrajdt[:, :, :, sim_params.nv:]

        loss = loss_function(traj, dtrajdt, alpha)
        loss.backward()
        # sim_params.ntime = optimal_time(sim_params.ntime, max_time, dt, loss_function, x_init, dyn_system, loss)
        optimizer.step()
        schedule_lr(optimizer, iteration, 20)
        wandb.log({'epoch': iteration+1, 'loss': loss.item()})

        print(f"Epochs: {iteration}, Loss: {loss.item()}, lr: {get_lr(optimizer)}, time: {sim_params.ntime} \n")

        selection = random.randint(0, sim_params.nsim - 1)

        if iteration % 25 == 0:
            ax1 = plt.subplot(122)
            ax2 = plt.subplot(121)
            ax1.clear()
            ax2.clear()
            for i in range(0, sim_params.nsim, 30):
                import numpy as np
                selection = random.randint(0, sim_params.nsim - 1)
                ax1.margins(0.05)  # Default margin is 0.05, value 0 means fit
                ax1.plot(traj[:, selection, 0, 0].cpu().detach())
                ax1.plot(traj[:, selection, 0, 1].cpu().detach())
                ax1.set_title('Pos')
                ax2 = plt.subplot(221)
                ax2.margins(0.05)  # Values >0.0 zoom out
                ax2.plot(acc[:, selection, 0, 0].cpu().detach())
                ax2.plot(acc[:, selection, 0, 1].cpu().detach())
                ax2.set_title('Acc')
                traj_dcp_mj = transform_coordinates_dcp(traj)
                ren.render(traj_dcp_mj[:, 0, 0, :dcp_params.nq].cpu().detach().numpy())


        iteration += 1
        plt.pause(0.001)

    model_scripted = torch.save(dyn_system.value_func.to('cpu'), f'{log}.pt')  # Export to TorchScript
    input()
