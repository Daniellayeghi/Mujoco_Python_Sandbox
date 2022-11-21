import math
import numpy as np
from utilities.mujoco_torch import mujoco, torch_mj_set_attributes, SimulationParams

import torch
import torch.nn.functional as Func
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utilities import mujoco_torch
from utilities.torch_utils import tensor_to_np
from utilities.torch_device import device
from utilities.data_utils import DataParams
from net_utils_torch import LayerInfo
from utilities.mujoco_torch import torch_mj_inv, torch_mj_set_attributes, torch_mj_detach, torch_mj_attach
import mujoco

use_cuda = torch.cuda.is_available()


# class PointMassData:
#     def __init__(self, sim_params: SimulationParams, q_init: torch.Tensor, qd_init: torch.Tensor):
#         self.q = q_init.clone().to(device)
#         self.qd = qd_init.clone().to(device)
#         self.qdd = torch.zeros(sim_params.nsim, 1, sim_params.nee).to(device)
#         self.sim_params = sim_params
#
#     def get_x(self):
#         return torch.cat((self.q, self.qd), 2).view(self.sim_params.nsim, 1, self.sim_params.nqv).clone()
#
#     def get_xd(self):
#         return torch.cat((self.qd, self.qdd), 2).view(self.sim_params.nsim, 1, self.sim_params.nqv).clone()
#
#     def get_xxd(self):
#         return torch.cat((self.q, self.qd, self.qdd), 2).view(self.sim_params.nsim, 1, self.sim_params.nqva).clone()
#
#     def detach(self):
#         self.q.detach()
#         self.qd.detach()
#         self.qdd.detach()
#
#     def attach(self):
#         self.q.requires_grad_()
#         self.qd.requires_grad_()
#         self.qdd.requires_grad_()


def decomp_x(x, sim_params: SimulationParams):
    return x[:, :, 0:sim_params.nq].clone(), x[:, :, sim_params.nq:].clone()


def decomp_xd(xd, sim_params: SimulationParams):
    return xd[:, :, 0:sim_params.nv].clone(), xd[:, :, sim_params.nv:].clone()


def ode_solve(zi, ti, tf, dt, dfdt):
    steps = math.ceil((abs(ti - tf)/dt).max().item())
    z, t = zi, ti
    for step in range(steps):
        z = z + dfdt(z, t) * dt
        t = t + dt

    return z


# def ode_solve(z0, t0, t1, f):
#     """
#     Simplest Euler ODE initial value solver
#     """
#     h_max = 0.01
#     n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())
#
#     h = (t1 - t0)/n_steps
#     t = t0
#     z = z0
#
#     for i_step in range(n_steps):
#         z = z + h * f(z, t)
#         t = t + h
#     return z


class ODEF(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        """Compute f and a df/dz, a df/dp, a df/dt"""
        batch_size = z.shape[0]

        out = self.forward(z, t)

        a = grad_outputs
        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,), (z, t) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )
        # grad method automatically sums gradients for batch items, we have to expand them back
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size
        return out, adfdz, adfdt, adfdp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)


class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):
        assert isinstance(func, ODEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)

        with torch.no_grad():
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0
            for i_t in range(time_len - 1):
                z0 = ode_solve(z0, t[i_t], t[i_t+1], 0.01, func)
                z[i_t+1] = z0

        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        return z

    @staticmethod
    def backward(ctx, dLdz):
        """
        dLdz shape: time_len, batch_size, *z_shape
        """
        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i, t_i):
            """
            tensors here are temporal slices
            t_i - is tensor with size: bs, 1
            aug_z_i - is tensor with size: bs, n_dim*2 + n_params + 1
            """
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2*n_dim]  # ignore parameters and time

            # Unflatten z and a
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)
            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)
                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)  # bs, *z_shape
                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)
                adfdt = adfdt.to(z_i) if adfdt is not None else torch.zeros(bs, 1).to(z_i)

            # Flatten f and adfdz
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim)
            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)

        dLdz = dLdz.view(time_len, bs, n_dim)  # flatten dLdz for convenience
        with torch.no_grad():
            ## Create placeholders for output gradients
            # Prev computed backwards adjoints to be adjusted by direct gradients
            adj_z = torch.zeros(bs, n_dim).to(dLdz)
            adj_p = torch.zeros(bs, n_params).to(dLdz)
            # In contrast to z and p we need to return gradients for all times
            adj_t = torch.zeros(time_len, bs, 1).to(dLdz)

            for i_t in range(time_len-1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                f_i = func(z_i, t_i).view(bs, n_dim)

                # Compute direct gradients
                dLdz_i = dLdz[i_t]
                dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

                # Adjusting adjoints with direct gradients
                adj_z += dLdz_i
                adj_t[i_t] = adj_t[i_t] - dLdt_i

                # Pack augmented variable
                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z), adj_t[i_t]), dim=-1)

                # Solve augmented system backwards
                aug_ans = ode_solve(aug_z, t_i, t[i_t-1], augmented_dynamics)

                # Unpack solved backwards augmented system
                adj_z[:] = aug_ans[:, n_dim:2*n_dim]
                adj_p[:] += aug_ans[:, 2*n_dim:2*n_dim + n_params]
                adj_t[i_t-1] = aug_ans[:, 2*n_dim + n_params:]

                del aug_z, aug_ans

            ## Adjust 0 time adjoint with direct gradients
            # Compute direct gradients
            dLdz_0 = dLdz[0]
            dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

            # Adjust adjoints
            adj_z += dLdz_0
            adj_t[0] = adj_t[0] - dLdt_0
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None


class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        self.func = func

    def forward(self, z0, t=Tensor([0., 1.]), return_whole_sequence=False):
        t = t.to(z0)
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func)
        if return_whole_sequence:
            return z
        else:
            return z[-1]


class ValueFunction(nn.Module):
    def __init__(self, W):
        super(ValueFunction, self).__init__()
        self.lin = nn.Linear(2, 2, bias=False)
        self.lin.weight = nn.Parameter(W)

    def forward(self, x, t):
        return self.lin(x)


class DynamicalSystem(ODEF):
    def __init__(self, value_function, loss, sim_params: SimulationParams):
        super(DynamicalSystem, self).__init__()
        self.value_func = value_function
        self.loss_func = loss
        self.sim_params = sim_params
        self.nsim = sim_params.nsim
        # self.point_mass = PointMassData(sim_params)

    def project(self, x, t):
        q, v = decomp_x(x, self.sim_params)
        xd = torch.cat((v, torch.zeros_like(v)), 2)
        # x_xd = torch.cat((q, v, torch.zeros_like(v)), 2)

        def dvdx(x, t, value_net):
            with torch.set_grad_enabled(True):
                x = x.detach().requires_grad_(True)
                value = value_net(x, t).requires_grad_()
                dvdx = torch.autograd.grad(
                    value, x, grad_outputs=torch.ones_like(value), create_graph=True, only_inputs=True
                )[0]
                return dvdx

        # Vx = torch.randn((1,1,2)).to(device)
        Vx = dvdx(x, t, self.value_func)
        norm = (Vx ** 2).sum(dim=2).view(self.nsim, 1, 1)
        unnorm_porj = Func.relu((Vx @ xd.mT) + self.loss_func(x))
        xd_trans = - (Vx / norm) * unnorm_porj
        return xd_trans[:, :, sim_params.nv:].view(1, 1, sim_params.nv)

    def dfdt(self, x, t):
        xd = self.project(x, t)
        v = x[:, :, -1].view(1, 1, sim_params.nv).clone()
        a = xd[:, :, -1].view(1, 1, sim_params.nv).clone()
        return torch.cat((v, a), 2)

    def forward(self, x, t):
        xd = self.dfdt(x, t)
        return xd


class LinearODEF(ODEF):
    def __init__(self, W):
        super(LinearODEF, self).__init__()
        self.lin = nn.Linear(2, 2, bias=False)
        self.lin.weight = nn.Parameter(W)

    def forward(self, x, t):
        return self.lin(x)


class RandomLinearODEF(LinearODEF):
    def __init__(self):
        super(RandomLinearODEF, self).__init__(torch.randn(2, 2)/2.)
        print("Done!!")


if __name__ == "__main__":
    m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml")
    d = mujoco.MjData(m)
    sim_params = SimulationParams(3, 2, 1, 1, 1, 1, 1)
    Q = torch.diag(torch.Tensor([1, .001])).view(sim_params.nsim, 2, 2).to(device)
    Qf = torch.diag(torch.Tensor([1, .5])).view(sim_params.nsim, 2, 2).to(device)

    def loss_func(x: torch.Tensor):
        return x @ Q @ x.mT

    def batch_loss(x: torch.Tensor):
        t, nsim, r, c = x.shape
        x_run = x[0:-1, :, :, :].view(t-1, r, c).clone()
        x_final = x[-1, :, :, :].view(1, r, c).clone()
        l_running = torch.sum(x_run @ Q @ x_run.mT, 0).squeeze()
        l_terminal = (x_final @ Qf @ x_final.mT).squeeze()

        return l_running + l_terminal


    q_init, qd_init = torch.randn((sim_params.nsim, 1, 1*sim_params.nee)), torch.zeros((sim_params.nsim, 1, 1*sim_params.nee))
    x_init = torch.cat((q_init, qd_init), 2).to(device)
    value_func = ValueFunction(torch.randn((2, 2))).to(device)
    dyn_system = DynamicalSystem(value_func, loss_func, sim_params).to(device)
    neural_ode = NeuralODE(RandomLinearODEF()).to(device)
    time = torch.linspace(0, 2, 21).to(device)

    ode_trained = NeuralODE(RandomLinearODEF())

    data_in = torch.Tensor([1, 0])
    time = torch.linspace(0, 2, 21)

    for i in range(100):
        x_out = ode_trained(data_in, time, return_whole_sequence=True)
        print(x_out)

    optimizer = torch.optim.Adam(neural_ode.parameters(), lr=0.01)

    epochs = range(100)
    for e in epochs:
        traj = neural_ode(x_init, time, return_whole_sequence=True)
        loss = batch_loss(traj)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
