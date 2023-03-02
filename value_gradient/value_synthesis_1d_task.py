import argparse
import mujoco
from neural_value_synthesis_diffeq import *
from utilities.torch_utils import save_models
from utilities.mujoco_torch import torch_mj_set_attributes, SimulationParams
from torchdiffeq import odeint_adjoint as odeint


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


if __name__ == "__main__":
    m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml")
    d = mujoco.MjData(m)
    sim_params = SimulationParams(3, 2, 1, 1, 1, 1, 50, 501, 0.01)
    prev_cost, diff, iteration, tol, max_iter, step_size = 0, 100.0, 1, 0.5, 100, 1.07
    Q = torch.diag(torch.Tensor([1, .01])).repeat(sim_params.nsim, 1, 1).to(device) * 10
    R = torch.diag(torch.Tensor([0.5])).repeat(sim_params.nsim, 1, 1).to(device)
    Qf = torch.diag(torch.Tensor([1, 0.01])).repeat(sim_params.nsim, 1, 1).to(device) * 100
    torch_mj_set_attributes(m, sim_params)

    def loss_func(x: torch.Tensor):
        return x @ Q @ x.mT

    def batch_state_loss(x: torch.Tensor):
        t, nsim, r, c = x.shape
        x_run = x[0:-1, :, :, :].view(t-1, nsim, r, c).clone()
        x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
        l_running = torch.sum(x_run @ Q @ x_run.mT, 0).squeeze()
        l_terminal = (x_final @ Qf @ x_final.mT).squeeze()

        return l_running + l_terminal


    def inv_dynamics_reg(acc: torch.Tensor, alpha):
        u_batch = acc
        loss = u_batch @ R @ u_batch.mT
        return torch.sum(loss, 0)


    def backup_loss(x: torch.Tensor):
        t, nsim, r, c = x.shape
        x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
        x_init = x[0, :, :, :].view(1, nsim, r, c).clone()
        value_final = nn_value_func((sim_params.ntime - 1) * 0.01, x_final).squeeze()
        value_init = nn_value_func(0, x_init).squeeze()

        return -value_init + value_final


    def batch_inv_dynamics_loss(acc, alpha):
        l_ctrl = inv_dynamics_reg(acc, alpha)
        return l_ctrl


    def loss_function(x, acc, alpha=1):
        l_ctrl, l_state, l_bellman = batch_inv_dynamics_loss(acc, alpha), batch_state_loss(x), backup_loss(x)
        loss = torch.mean(l_ctrl + l_state + l_bellman)
        return torch.maximum(loss, torch.zeros_like(loss))


    class NNValueFunction(nn.Module):
        def __init__(self, n_in):
            super(NNValueFunction, self).__init__()

            self.nn = nn.Sequential(
                nn.Linear(n_in, 16, bias=False),
                nn.Softplus(),
                nn.Linear(16, 1, bias=False),
            )

            def init_weights(net):
                if type(net) == nn.Linear:
                    torch.nn.init.xavier_uniform(net.weight)

            self.nn.apply(init_weights)

        def forward(self, t, x):
            return self.nn(x)

    x_encoder = lambda x: x

    S_init = torch.FloatTensor(sim_params.nqv, sim_params.nqv).uniform_(0, 5).to(device)
    # S_init = torch.Tensor([[1.7, 1], [1, 1.7]]).to(device)
    nn_value_func = NNValueFunction(sim_params.nqv).to(device)
    dyn_system = ProjectedDynamicalSystem(nn_value_func, loss_func, sim_params, encoder=x_encoder).to(device)
    time = torch.linspace(0, (sim_params.ntime - 1) * 0.01, sim_params.ntime).to(device)
    optimizer = torch.optim.Adam(dyn_system.parameters(), lr=5e-3)

    q_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nq).uniform_(-1, 1) * 5.5
    qd_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nq).uniform_(-1, 1) * 2
    # q_init = torch.ones((sim_params.nsim, 1, 1 * sim_params.nee))
    # qd_init = torch.zeros((sim_params.nsim, 1, 1 * sim_params.nee))
    x_init = torch.cat((q_init, qd_init), 2).to(device)
    pos_arr = torch.linspace(-10, 10, 100).to(device)
    vel_arr = torch.linspace(-10, 10, 100).to(device)
    f_mat = torch.zeros((100, 100)).to(device)
    [X, Y] = torch.meshgrid(pos_arr.squeeze().cpu(), vel_arr.squeeze().cpu())

    while diff > tol or iteration > max_iter:
        optimizer.zero_grad()
        traj = odeint(dyn_system, x_init, time, method='euler')
        acc = compose_acc(traj[:, :, :, sim_params.nv:].clone(), 0.01)
        loss = loss_function(traj, acc)
        # xxd = compose_xxd(traj, acc)
        # loss = batch_state_ctrl_loss(traj, xxd)

        # dyn_system.step *= step_size
        loss.backward()
        optimizer.step()

        print(f"Epochs: {iteration}, Loss: {loss.item()}")
        iteration += 1

        if True:
            with torch.no_grad():
                plot_2d_funcition(pos_arr, vel_arr, [X, Y], f_mat, nn_value_func, trace=traj, contour=True)

    save_models("./neural_value", nn_value_func)
    plt.show()
