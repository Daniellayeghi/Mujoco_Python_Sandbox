import random
import torch
from models import DoubleCartpole, ModelParams, init_fig_dcp, animate_double_cartpole
from neural_value_synthesis_diffeq import *
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from utilities.mujoco_torch import SimulationParams

sim_params = SimulationParams(9,6,3,3,1,1,80,400,0.01)
dcp_params = ModelParams(3, 3, 1, 6, 6)
prev_cost, diff, tol, max_iter, alpha, dt, n_bins, discount, step, mode = 0, 100.0, 0, 500, .5, 0.001, 3, 1.025, 15, 'hjb'
Q = torch.diag(torch.Tensor([.5, .2, .2, 0.0001, 0.0001, 0.0001])).repeat(sim_params.nsim, 1, 1).to(device)
R = torch.diag(torch.Tensor([0.0001])).repeat(sim_params.nsim, 1, 1).to(device)
Qf = torch.diag(torch.Tensor([5, 300, 300, 10, 10, 10])).repeat(sim_params.nsim, 1, 1).to(device)
lambdas = torch.ones((sim_params.ntime, sim_params.nsim, 1, 1))
cartpole = DoubleCartpole(sim_params.nsim, dcp_params, device)


def build_discounts(lambdas: torch.Tensor, discount: float):
    for i in range(lambdas.shape[0]):
        lambdas[i, :, :, :] *= (discount)**i

    return lambdas.clone()


def state_encoder(x: torch.Tensor):
    b, r, c = x.shape
    x = x.reshape((b, r*c))
    qc, qp, v = x[:, 0].clone().unsqueeze(1), x[:, 1:sim_params.nq].clone(), x[:, sim_params.nq:].clone()
    qp = torch.cos(qp) - 1
    return torch.cat((qc, qp, v), 1).reshape((b, r, c))


def batch_state_encoder(x: torch.Tensor):
    t, b, r, c = x.shape
    x = x.reshape((t*b, r*c))
    qc, qp, v = x[:, 0].clone().unsqueeze(1), x[:, 1:sim_params.nq].clone(), x[:, sim_params.nq:].clone()
    qp = torch.cos(qp) - 1
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
            nn.Linear(n_in, 64),
            nn.Softplus(beta=5),
            nn.Linear(64, 64),
            nn.Softplus(beta=5),
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


def loss_func(x: torch.Tensor, t):
    x = state_encoder(x)
    return x @ Q @ x.mT * (discount ** (t*sim_params.dt))


def backup_loss(x: torch.Tensor):
    t, nsim, r, c = x.shape
    x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
    factor = lambdas[-1, :, :, :].view(1, nsim, 1, 1).clone()
    x_final_w = batch_state_encoder(x_final)
    l_running = (x_final_w @ Q @ x_final_w.mT) * factor
    l_running = torch.sum(l_running, 0).squeeze()
    value = nn_value_func(0, x_final_w).squeeze()
    return torch.mean(torch.square(value - l_running))


def lyapounov_goal_loss(x: torch.Tensor):
    t, nsim, r, c = x.shape
    x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
    x_final_w = batch_state_encoder(x_final)
    value = nn_value_func(0, x_final_w).squeeze()
    return torch.mean(value)


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
    x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
    l_running = (x @ Q @ x.mT) * lambdas
    l_running = torch.sum(l_running, 0).squeeze()
    l_terminal = (x_final @ Qf @ x_final.mT).squeeze() * 0

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
    C = cartpole._Tbias(x_reshape).reshape((x.shape[0], x.shape[1], 1, sim_params.nv))
    u_batch = (M @ acc.mT).mT + C
    return torch.mean(torch.sum(u_batch @ torch.linalg.inv(M) @ u_batch.mT, 0).squeeze()) * 0.00001


def loss_function_bellman(x, acc, alpha=1):
    l_ctrl, l_state, l_bellman, l_dyn = batch_inv_dynamics_loss(x, acc, alpha), batch_state_loss(x), backup_loss(x), batch_dynamics_loss(x, acc, alpha)
    print(f"loss ctrl {l_ctrl}, loss state {l_state}, loss bellman {l_bellman}, loss dynamics {l_dyn}, alpha {alpha}")
    return l_ctrl + l_state + l_bellman + l_dyn


def loss_function_lyapounov(x, acc, alpha=1):
    l_ctrl, l_state, l_lyap = batch_inv_dynamics_loss(x, acc, alpha), batch_state_loss(x), lyapounov_goal_loss(x)
    print(f"loss ctrl {l_ctrl}, loss state {l_state}, loss bellman {l_lyap}, alpha {alpha}")
    return l_ctrl + l_state + l_lyap


thetas = torch.linspace(torch.pi - 0.6, torch.pi + 0.6, n_bins)
mid_point = int(len(thetas)/2) + len(thetas) % 2
dyn_system = ProjectedDynamicalSystem(
    nn_value_func, loss_func, sim_params, encoder=state_encoder, dynamics=cartpole, mode=mode, step=step
).to(device)
time = torch.linspace(0, (sim_params.ntime - 1) * dt, sim_params.ntime).to(device)
one_step = torch.linspace(0, dt, 2).to(device)
optimizer = torch.optim.AdamW(dyn_system.parameters(), lr=3e-2, amsgrad=True)
lambdas = build_discounts(lambdas, discount).to(device)

fig_3, p, r, width, height = init_fig_dcp(0)

log = f"m-{mode}_d-{discount}_s-{step}"


if __name__ == "__main__":
    qp_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nq-1).uniform_(torch.pi - 0.3, torch.pi + 0.3) * 1
    qc_init = torch.FloatTensor(sim_params.nsim, 1, 1).uniform_(0, 0) * 1
    qd_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nv).uniform_(0, 0) * 1
    x_init = torch.cat((qc_init, qp_init, qd_init), 2).to(device)
    iteration = 0

    def schedule_lr(optimizer, epoch, rate):
        pass
        if epoch % rate == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= (0.75 * (0.95 ** (epoch / rate)))

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    while iteration < max_iter:
        optimizer.zero_grad()

        traj = odeint(
            dyn_system, x_init, time, method='euler',
            options=dict(step_size=dt), adjoint_atol=1e-9, adjoint_rtol=1e-9
        )

        acc = compose_acc(traj, dt)
        loss = loss_function_bellman(traj, acc, alpha)
        loss.backward()
        optimizer.step()
        schedule_lr(optimizer, iteration, 60)

        print(f"Epochs: {iteration}, Loss: {loss.item()}, iteration: {iteration % 10}, lr: {get_lr(optimizer)}")

        selection = random.randint(0, sim_params.nsim - 1)

        if iteration % 20 == 0 and iteration != 0:
            fig_1 = plt.figure(1)
            for i in range(sim_params.nsim):
                qpole = traj[:, i, 0, 1].cpu().detach()
                qdpole = traj[:, i, 0, 3].cpu().detach()
                plt.plot(qpole, qdpole)

            plt.pause(0.001)
            fig_2 = plt.figure(2)
            ax_2 = plt.axes()
            plt.plot(traj[:, selection, 0, 0].cpu().detach())
            plt.pause(0.001)
            ax_2.set_title(loss.item())

            for i in range(0, sim_params.nsim, 10):
                selection = random.randint(0, sim_params.nsim - 1)
                cart = traj[:, selection, 0, 0].cpu().detach().numpy()
                pole1 = traj[:, selection, 0, 1].cpu().detach().numpy()
                pole2 = traj[:, selection, 0, 2].cpu().detach().numpy()

                animate_double_cartpole(cart, pole1, pole2, fig_3, p, r, width, height, skip=2)

            fig_1.clf()
            fig_2.clf()

        iteration += 1

    model_scripted = torch.save(dyn_system.value_func.to('cpu'), f'{log}.pt')  # Export to TorchScript
    input()