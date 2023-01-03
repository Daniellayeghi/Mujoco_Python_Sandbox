import argparse
import torch
import torch.nn.functional as Func
from torch import nn
import matplotlib.pyplot as plt
from utilities.torch_device import device
from utilities.mujoco_torch import SimulationParams

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


def compose_xxd(x, acc):
    return torch.cat((x, acc), dim=3)


def compose_acc(x, dt):
    ntime, nsim, r, c = x.shape
    v = x[:, :, :, int(c/2):].clone()
    acc = torch.diff(v, dim=0) / dt
    acc_null = torch.zeros_like((acc[0, :, :, :])).view(1, nsim, r, int(c/2))
    return torch.cat((acc, acc_null), dim=0)


class ProjectedDynamicalSystem(nn.Module):
    def __init__(self, value_function, loss, sim_params: SimulationParams, dynamics=None):
        super(ProjectedDynamicalSystem, self).__init__()
        self.value_func = value_function
        self.loss_func = loss
        self.sim_params = sim_params
        self.nsim = sim_params.nsim
        self._dynamics = dynamics
        self.step =  0.015#nn.Parameter(torch.tensor(0.015))

        if dynamics is None:
            def dynamics(x, xd):
                v = x[:, :, self.sim_params.nq:].view(self.sim_params.nsim, 1, self.sim_params.nv).clone()
                a = xd.clone()
                return torch.cat((v, a), 2)
            self._dynamics = dynamics

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
        xd_trans = - (Vx / norm) * unnorm_porj * 5
        return xd_trans[:, :, self.sim_params.nv:].view(self.sim_params.nsim, 1, self.sim_params.nv)

    def dfdt(self, t, x):
        # TODO: Either the value function is a function of just the actuation space e.g. the cart or it takes into
        # TODO: the main difference is that the normalised projection is changed depending on what is used
        xd = self.project(t, x)
        return self._dynamics(x, xd)

        # v = x[:, :, self.sim_params.nq:].view(self.sim_params.nsim, 1, self.sim_params.nv).clone()
        # a = xd.clone()
        # return torch.cat((v, a), 2)

    def forward(self, t, x):
        xd = self.dfdt(t, x)
        return xd
