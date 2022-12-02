import math
import mujoco
import torch.linalg
import matplotlib.pyplot as plt

from neural_value_synthesis_diffeq import *
from utilities.torch_utils import save_models
from utilities.mujoco_torch import torch_mj_set_attributes, SimulationParams, torch_mj_inv


def plot_2d_funcition(xs: torch.Tensor, ys: torch.Tensor, xy_grid, f_mat, func, loss, trace=None, contour=True):
    assert len(xs) == len(ys)

    def plot_state(xs: torch.Tensor, ax):
        _, nsim, _, c = xs.shape
        nq = int(c/2)
        for i in range(nsim):
            x = xs[:, i, :, :nq].squeeze()
            y = xs[:, i, :, nq:].squeeze()
            ax.plot(x, y)

    plt.clf()
    ax = plt.axes()

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            arr = [x.item(), 0, y.item(), 0]
            in_tensor = torch.tensor(arr).view(1, 1, 4).float().to(device)
            f_mat[i, j] = func(0, in_tensor).detach().squeeze()

    [X, Y] = xy_grid
    f_mat = f_mat.cpu()

    if contour:
        ax.contourf(X, Y, f_mat, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    else:
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, f_mat, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    plot_state(trace, ax)
    ax.set_title(f'surface loss {loss}')
    ax.set_xlabel('Pos')
    ax.set_ylabel('Vel')
    plt.pause(0.001)


if __name__ == "__main__":
    m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator_sparse.xml")
    d = mujoco.MjData(m)
    sim_params = SimulationParams(3 * 2, 2 * 2, 1 * 2, 1 * 2, 1 * 2, 2, 50, 101)
    prev_cost, diff, iteration, tol, max_iter, step_size = 0, 100.0, 1, -1, 10000, 1.02
    Q = torch.diag(torch.Tensor([0, 1, 0, 1])).repeat(sim_params.nsim, 1, 1).to(device)
    Q_stop = torch.diag(torch.Tensor([0, 0, 1, 0])).repeat(sim_params.nsim, 1, 1).to(device)
    Qb = torch.diag(torch.Tensor([0, 1, 0, 1])).repeat(sim_params.nsim, 1, 1).to(device)
    R = torch.diag(torch.Tensor([0, 10000])).repeat(sim_params.nsim, 1, 1).to(device)
    Qf = torch.diag(torch.Tensor([0, 1, 0, 1])).repeat(sim_params.nsim, 1, 1).to(device)
    torch_mj_set_attributes(m, sim_params)

    def loss_func(x: torch.Tensor):
        x_a = x[:, :, :sim_params.nq-1].clone()
        x_u = x[:, :, sim_params.nq-1:sim_params.nq].clone()
        l_dist = torch.cdist(x_a, x_u)
        l_dist_norm = l_dist / torch.max(l_dist)
        x_task = x.view(x.shape).clone()
        l_task = (x_task @ Q @ x_task.mT) * 1
        l_stop = (x_task @ Q_stop @ x_task.mT) * 1
        return ((l_stop + l_task) * torch.exp(l_dist_norm) + l_dist)

    def batch_distance_loss(x: torch.Tensor):
        t, nsim, r, c = x.shape
        x_run_a = x[0:-1, :, :, :sim_params.nq-1].view(t-1, nsim, r, 1).clone()
        x_run_u = x[0:-1, :, :, sim_params.nq-1:sim_params.nq].view(t-1, nsim, r, 1).clone()
        l_dist = torch.cdist(x_run_a, x_run_u).view(t-1, nsim, r, 1) * 5
        return l_dist

    def batch_state_terminal_cst(x: torch.Tensor):
        t, nsim, r, c = x.shape
        x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
        l_terminal = (x_final @ Qf @ x_final.mT) * 1
        return l_terminal


    def batch_state_running_cst(x: torch.Tensor):
        t, nsim, r, c = x.shape
        x_run = x[0:-1, :, :, :].view(t-1, nsim, r, c).clone()
        l_task = x_run @ Qb @ x_run.mT * 1
        l_stop = x_run @ Q_stop @ x_run.mT * 1
        return l_task + l_stop

        # # Assemble losses
        # l_dist_norm = l_dist / torch.max(l_dist)
        # l_task = x_run @ Qb @ x_run.mT * 1
        # l_stop = x_run @ Q_stop @ x_run.mT * 1
        # l_running = torch.sum((l_stop + l_task) * torch.exp(l_dist_norm) + l_dist, dim=0).view(1, nsim, r, 1)
        # l_terminal = (x_final @ Qf @ x_final.mT) * 1
        # return torch.mean(l_running + l_terminal, 1).squeeze()

    def batch_ctrl_cst(xxd: torch.Tensor, cio):
        t, nsim, r, c = xxd.shape
        qfrcs = torch_mj_inv.apply(xxd)
        # loss = torch.sum(qfrcs @ R @ qfrcs.mT, dim=0).view(1, nsim, r, 1)
        loss_act = (qfrcs[:, :, :, :sim_params.nv-1] * 0).square_().clone()
        loss_uact = (qfrcs[:, :, :, sim_params.nv-1:] * cio).square_().clone()
        loss = torch.sum(loss_uact + loss_act, dim=0).view(1, nsim, r, 1)
        return torch.mean(loss, 1).squeeze()

    def batch_state_ctrl_loss(x, xxd, cio):
        l_dist = batch_distance_loss(x)
        l_run = torch.sum(batch_state_running_cst(x), 0)
        l_term = batch_state_terminal_cst(x)
        l_ctrl = torch.sum(batch_ctrl_cst(xxd, cio * l_dist), 0)


    class NNValueFunction(nn.Module):
        def __init__(self, n_in):
            super(NNValueFunction, self).__init__()

            self.nn = nn.Sequential(
                nn.Linear(n_in, 4, bias=False),
                nn.Softplus(),
                nn.Linear(4, 1, bias=False)
            )

            def init_weights(net):
                if type(net) == nn.Linear:
                    torch.nn.init.xavier_uniform(net.weight)

            self.nn.apply(init_weights)

        def forward(self, t, x):
            return self.nn(x)

    # Initialise the network
    S_init = torch.FloatTensor(sim_params.nqv, sim_params.nqv).uniform_(0, 5).to(device)
    nn_value_func = NNValueFunction(sim_params.nqv).to(device)
    dyn_system = DynamicalSystem(nn_value_func, loss_func, sim_params).to(device)
    optimizer = torch.optim.Adam(dyn_system.parameters(), lr=3e-2)

    # Assemble initial conditions
    q_init_act = torch.FloatTensor(sim_params.nsim, 1, 1).uniform_(-7.5, 7.5)
    q_init_uact = torch.zeros(sim_params.nsim, 1, 1)
    q_init = torch.cat((q_init_act, q_init_uact), 2)
    qd_init_act = torch.FloatTensor(sim_params.nsim, 1, 1).uniform_(-5, 5)
    qd_init_uact = torch.zeros((sim_params.nsim, 1, 1))
    qd_init = torch.cat((qd_init_act, qd_init_uact), 2)
    x_init = torch.cat((q_init, qd_init), 2).to(device)

    # Assemble function grid
    pos_arr = torch.linspace(-10, 10, 100).to(device)
    vel_arr = torch.linspace(-10, 10, 100).to(device)
    f_mat = torch.zeros((100, 100)).to(device)
    [X, Y] = torch.meshgrid(pos_arr.squeeze().cpu(), vel_arr.squeeze().cpu())
    time = torch.linspace(0, (sim_params.ntime - 1) * 0.01, sim_params.ntime).to(device)
    cios = torch.linspace(0, 10, 1001)

    while diff > tol and iteration < max_iter:
        optimizer.zero_grad()
        traj = odeint(dyn_system, x_init, time)
        acc = compose_acc(traj, 0.01)
        xxd = compose_xxd(traj, acc)
        loss = batch_state_ctrl_loss(traj, xxd, min(cios[iteration].item(), 1))
        dyn_system.step *= step_size
        diff = math.fabs(prev_cost - loss.item())
        prev_cost = loss.item()
        loss.backward()
        optimizer.step()

        for param in dyn_system.parameters():
            print(f"\n{param}\n")

        print(f"Epochs: {iteration}, Loss: {loss.item()} Stepping with {dyn_system.step}")
        iteration += 1

        if True:
            with torch.no_grad():
                plot_2d_funcition(
                    pos_arr, vel_arr, [X, Y], f_mat, nn_value_func, loss.item(), trace=traj, contour=True
                )

    save_models("./neural_value", nn_value_func)
    plt.show()
