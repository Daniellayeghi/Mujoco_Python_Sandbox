import math
import argparse
import numpy as np
from utilities.mujoco_torch import mujoco, torch_mj_set_attributes, SimulationParams
import torch
import torch.nn.functional as Func
from torch import Tensor
from torch import nn
import matplotlib.pyplot as plt
from utilities.torch_device import device
import mujoco


parser = argparse.ArgumentParser('Neural Value Synthesis demo')
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
    print(f"Using the Adjoint method")
else:
    from torchdiffeq import odeint


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


def decomp_x(x, sim_params: SimulationParams):
    return x[:, :, 0:sim_params.nq].clone(), x[:, :, sim_params.nq:].clone()


def decomp_xd(xd, sim_params: SimulationParams):
    return xd[:, :, 0:sim_params.nv].clone(), xd[:, :, sim_params.nv:].clone()


class LinValueFunction(nn.Module):
    """
    Value function is J = xSx
    """
    def __init__(self, n_in, Sinit):
        super(LinValueFunction, self).__init__()
        self.S = nn.Linear(2, 2, bias=-False)
        self.S.weight = nn.Parameter(Sinit)

    def forward(self, t: float, x: torch.Tensor):
        return self.S(x) @ x.mT


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


class DynamicalSystem(nn.Module):
    def __init__(self, value_function, loss, sim_params: SimulationParams):
        super(DynamicalSystem, self).__init__()
        self.value_func = value_function
        self.loss_func = loss
        self.sim_params = sim_params
        self.nsim = sim_params.nsim
        self.step = 5
        # self.point_mass = PointMassData(sim_params)

    def project(self, t, x):
        q, v = decomp_x(x, self.sim_params)
        xd = torch.cat((v, torch.zeros_like(v)), 2)
        # x_xd = torch.cat((q, v, torch.zeros_like(v)), 2)

        def dvdx(t, x, value_net):
            with torch.set_grad_enabled(True):
                x = x.detach().requires_grad_(True)
                value = value_net(t, x).requires_grad_()
                dvdx = torch.autograd.grad(
                    value, x, grad_outputs=torch.ones_like(value), create_graph=True, only_inputs=True
                )[0]
                return dvdx

        Vx = dvdx(t, x, self.value_func)
        norm = ((Vx @ Vx.mT) + 1e-6).sqrt().view(self.nsim, 1, 1)
        unnorm_porj = Func.relu((Vx @ xd.mT) + self.step * self.loss_func(x))
        xd_trans = - (Vx / norm) * unnorm_porj
        return xd_trans[:, :, self.sim_params.nv:].view(self.sim_params.nsim, 1, self.sim_params.nv)

    def dfdt(self, t, x):
        # Fix a =
        xd = self.project(t, x)
        v = x[:, :, -1].view(self.sim_params.nsim, 1, self.sim_params.nv).clone()
        a = xd[:, :, -1].view(self.sim_params.nsim, 1, self.sim_params.nv).clone()
        return torch.cat((v, a), 2)

    def forward(self, t, x):
        xd = self.dfdt(t, x)
        return xd


def save_models(value_path: str, net: nn.Module):
    print("########## Saving Trace ##########")
    torch.save(net.state_dict(), value_path + ".pt")


if __name__ == "__main__":
    m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml")
    d = mujoco.MjData(m)
    sim_params = SimulationParams(3, 2, 1, 1, 1, 1, 40)
    Q = torch.diag(torch.Tensor([1, 1])).repeat(sim_params.nsim, 1, 1).to(device)
    R = torch.diag(torch.Tensor([0.5])).repeat(sim_params.nsim, 1, 1).to(device)
    Qf = torch.diag(torch.Tensor([1, 1])).repeat(sim_params.nsim, 1, 1).to(device)

    def loss_func(x: torch.Tensor):
        return x @ Q @ x.mT

    def batch_loss(x: torch.Tensor):
        t, nsim, r, c = x.shape
        x_run = x[0:-1, :, :, :].view(t-1, nsim, r, c).clone()
        x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
        l_running = torch.sum(x_run @ Q @ x_run.mT, 0).squeeze()
        l_terminal = (x_final @ Qf @ x_final.mT).squeeze()

        return torch.mean(l_running + l_terminal)

    S_init = torch.FloatTensor(sim_params.nqv, sim_params.nqv).uniform_(0, 5).to(device)
    # S_init = torch.Tensor([[1.7, 1], [1, 1.7]]).to(device)
    lin_value_func = LinValueFunction(sim_params.nqv, S_init).to(device)
    nn_value_func = NNValueFunction(sim_params.nqv).to(device)
    dyn_system = DynamicalSystem(nn_value_func, loss_func, sim_params).to(device)
    time = torch.linspace(0, 5, 501).to(device)
    optimizer = torch.optim.AdamW(dyn_system.parameters(), lr=3e-2)

    q_init = torch.FloatTensor(sim_params.nsim, 1, 1 * sim_params.nee).uniform_(-1, 1) * 5
    qd_init = torch.FloatTensor(sim_params.nsim, 1, 1 * sim_params.nee).uniform_(-1, 1) * 5
    # q_init = torch.ones((sim_params.nsim, 1, 1 * sim_params.nee))
    # qd_init = torch.zeros((sim_params.nsim, 1, 1 * sim_params.nee))
    x_init = torch.cat((q_init, qd_init), 2).to(device)
    pos_arr = torch.linspace(-10, 10, 100).to(device)
    vel_arr = torch.linspace(-10, 10, 100).to(device)
    f_mat = torch.zeros((100, 100)).to(device)
    [X, Y] = torch.meshgrid(pos_arr.squeeze().cpu(), vel_arr.squeeze().cpu())

    epochs, attempts = range(100), range(10)

    for e in epochs:
        optimizer.zero_grad()
        traj = odeint(dyn_system, x_init, time)
        loss = batch_loss(traj)
        # dyn_system.step /= (0.1 * loss.item())
        loss.backward()
        optimizer.step()

        for param in dyn_system.parameters():
            print(f"\n{param}\n")

        print(f"Epochs: {e}, Loss: {loss.item()}")
        # if e % 0 == 0:
        with torch.no_grad():
            plot_2d_funcition(pos_arr, vel_arr, [X, Y], f_mat, lin_value_func, trace=traj, contour=True)

    save_models("./neural_value", lin_value_func)
    plt.show()
