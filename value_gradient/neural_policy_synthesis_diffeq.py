import argparse
import torch
import torch.nn.functional as Func
from torch import nn
import matplotlib.pyplot as plt
from utilities.torch_device import device
from utilities.mujoco_torch import SimulationParams



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


def compose_xxd(x, acc):
    return torch.cat((x, acc), dim=3)


def compose_acc(x, dt):
    ntime, nsim, r, c = x.shape
    v = x[:, :, :, int(c / 2):].clone()
    acc = torch.diff(v, dim=0) / dt
    acc_null = torch.zeros_like((acc[0, :, :, :])).view(1, nsim, r, int(c / 2))
    return torch.cat((acc, acc_null), dim=0)


class ControlledDynamicalSystem(nn.Module):
    def __init__(self, policy_func, sim_params: SimulationParams, dynamics=None, encoder=None):
        super(ControlledDynamicalSystem, self).__init__()
        self.policy_func = policy_func
        self.sim_params = sim_params
        self.nsim = sim_params.nsim
        self._dynamics = dynamics
        self._encoder = encoder

    def control(self, t, x):
        x_enc = self._encoder(x)
        return self.policy_func(t, x_enc)

    def dfdt(self, t, x):
        # TODO: Either the value function is a function of just the actuation space e.g. the cart or it takes into
        # TODO: the main difference is that the normalised projection is changed depending on what is used
        acc = self.control(t, x)
        return self._dynamics(x, acc)

    def forward(self, t, x):
        xd = self.dfdt(t, x)
        return xd
