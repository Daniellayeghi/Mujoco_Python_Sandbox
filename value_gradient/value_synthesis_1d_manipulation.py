import math
import mujoco
import torch.linalg
import matplotlib.pyplot as plt

from neural_value_synthesis_diffeq import *
from utilities.torch_utils import save_models
from utilities.mujoco_torch import torch_mj_set_attributes, SimulationParams, torch_mj_inv


def plot_state(xs: torch.Tensor, sim_params: SimulationParams):
    nq = int(sim_params.nq/2)
    x1, x2 = xs[-1, :, :, :nq].squeeze().item(), xs[-1, :, :, nq:nq+1].squeeze().item()
    plt.clf()
    ax = plt.axes()
    ax.scatter([x1, x2], [0, 0], color=["red", "green"])
    plt.pause(0.001)


def plot_2d_funcition(xs: torch.Tensor, ys: torch.Tensor, xy_grid, f_mat, func, trace=None, contour=True):
    assert len(xs) == len(ys)
    trace = trace[:, :, :, :2].detach().clone().cpu()
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            arr = [x.item(), 0, y.item(), 0]
            in_tensor = torch.tensor(arr).view(1, 1, 4).float().to(device)
            f_mat[i, j] = func(0, in_tensor).detach().squeeze()

    [X, Y] = xy_grid
    f_mat = f_mat.cpu()
    plt.clf()
    ax = plt.axes()
    # if contour:
    #     # ax.contourf(X, Y, f_mat, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # else:
    #     ax = plt.axes(projection='3d')
    #     ax.plot_surface(X, Y, f_mat, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('surface')
    ax.set_xlabel('Pos')
    ax.set_ylabel('Vel')
    n_plots = trace.shape[1]
    for i in range(n_plots):
        ax.plot(trace[:, i, :, 0], trace[:, i, :, 1])
    plt.pause(0.001)


if __name__ == "__main__":
    m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator_sparse.xml")
    d = mujoco.MjData(m)
    sim_params = SimulationParams(3 * 2, 2 * 2, 1 * 2, 1 * 2, 1 * 2, 2, 1, 201)
    prev_cost, diff, iteration, tol, max_iter, step_size = 0, 100.0, 1, -1, 10000, 1.03
    Q = torch.diag(torch.Tensor([0, 1, 0, 1])).repeat(sim_params.nsim, 1, 1).to(device)
    Qb = torch.diag(torch.Tensor([1, 1, 1, 1])).repeat(sim_params.nsim, 1, 1).to(device)
    R = torch.diag(torch.Tensor([0.5])).repeat(sim_params.nsim, 1, 1).to(device)
    Qf = torch.diag(torch.Tensor([0, 1, 0, 1])).repeat(sim_params.nsim, 1, 1).to(device)
    torch_mj_set_attributes(m, sim_params)

    def loss_func(x: torch.Tensor):
        x_a = x[:, :, :sim_params.nq-1].clone()
        x_u = x[:, :, sim_params.nq-1:sim_params.nq].clone()
        loss_dist = torch.cdist(x_a, x_u)
        x_task = x.view(x.shape).clone()
        # loss_dist = torch.linalg.vector_norm(x_dist, dim=2).view(nsim, r, 1) * 10
        loss_task = (x_task @ Q @ x_task.mT) * 0
        return (loss_task * torch.exp(loss_dist) + loss_dist).squeeze()

    def batch_state_loss(x: torch.Tensor):
        t, nsim, r, c = x.shape
        x_run = x[0:-1, :, :, :].view(t-1, nsim, r, c).clone()
        x_run_a = x[0:-1, :, :, :sim_params.nq-1].view(t-1, nsim, r, 1).clone()
        x_run_u = x[0:-1, :, :, sim_params.nq-1:sim_params.nq].view(t-1, nsim, r, 1).clone()
        x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
        l_task = x_run @ Q @ x_run.mT * 0
        l_dist = torch.cdist(x_run_a, x_run_u).view(t-1, nsim, r, 1) * 100
        l_dist_norm = l_dist / torch.max(l_dist)
        l_running = torch.sum(l_task * torch.exp(l_dist_norm) + l_dist, dim=0).view(1, nsim, r, 1)
        l_terminal = (x_final @ Qf @ x_final.mT) * 0
        return torch.mean(l_running + l_terminal, 1).squeeze()

    def batch_ctrl_loss(xxd: torch.Tensor, discount):
        qfrcs = torch_mj_inv.apply(xxd)
        loss_act = (qfrcs[:, :, :, :sim_params.nv-1] * 0).square_().clone()
        loss_uact = (qfrcs[:, :, :, sim_params.nv-1:] * 10000).square_().clone()
        loss = torch.sum(loss_uact + loss_act, dim=0)
        return torch.mean(loss, 1)

    def batch_state_ctrl_loss(x, xxd, discount):
        return batch_state_loss(x) + 1 * batch_ctrl_loss(xxd, discount)

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
    optimizer = torch.optim.AdamW(dyn_system.parameters(), lr=3e-2)

    # Assemble initial conditions
    q_init_act = torch.FloatTensor(sim_params.nsim, 1, 1).uniform_(-2.5, -2)
    q_init_uact = torch.FloatTensor(sim_params.nsim, 1, 1).uniform_(-1.5, -1)
    q_init_uact = torch.zeros(sim_params.nsim, 1, 1)
    q_init = torch.cat((q_init_act, q_init_uact), 2)
    qd_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nq).uniform_(-0.0, 0) * 5
    qd_init = torch.zeros((sim_params.nsim, 1, sim_params.nv))
    x_init = torch.cat((q_init, qd_init), 2).to(device)

    # Assemble function grid
    pos_arr = torch.linspace(-10, 10, 100).to(device)
    vel_arr = torch.linspace(-10, 10, 100).to(device)
    f_mat = torch.zeros((100, 100)).to(device)
    [X, Y] = torch.meshgrid(pos_arr.squeeze().cpu(), vel_arr.squeeze().cpu())
    time = torch.linspace(0, (sim_params.ntime - 1) * 0.01, sim_params.ntime).to(device)
    cio = 100
    while diff > tol and iteration < max_iter:
        optimizer.zero_grad()
        traj = odeint(dyn_system, x_init, time)
        acc = compose_acc(traj, 0.01)
        xxd = compose_xxd(traj, acc)
        loss = batch_state_ctrl_loss(traj, xxd, cio)
        cio *= 1.02

        dyn_system.step *= step_size
        print(f"Stepping with {dyn_system.step}")

        diff = math.fabs(prev_cost - loss.item())
        prev_cost = loss.item()
        loss.backward()
        optimizer.step()

        for param in dyn_system.parameters():
            print(f"\n{param}\n")

        print(f"Epochs: {iteration}, Loss: {loss.item()}")
        iteration += 1

        if True:
            with torch.no_grad():
                # plot_2d_funcition(pos_arr, vel_arr, [X, Y], f_mat, nn_value_func, trace=traj, contour=True)
                plot_state(traj, sim_params)

    save_models("./neural_value", nn_value_func)
    plt.show()
