import numpy as np
import torch
from utilities import mujoco_torch
from utilities.torch_device import device
from utilities.data_utils import *
from networks import ValueFunction, MLP
from net_utils_torch import LayerInfo
import torch.nn.functional as Func
import mujoco


class PointMassData:
    def __init__(self, n_bodies):
        self.x_xd = torch.zeros((3, n_bodies)).to(device).requires_grad_()
        self.qpos = self.x_xd[0, :]
        self.qvel = self.x_xd[1, :]
        self.qacc = self.x_xd[2, :]


def step(data: PointMassData, dt):
    data.qvel = data.qvel + data.qacc * dt
    data.qpos = data.qpos + data.qvel * dt


# Data params
d_params = DataParams(3, 2, 1, 1, 1, 2, [1, 2], 16)

# Networks and optimizers
val_input, value_output = d_params.n_state, 1
layer_dims = [val_input, 32, 64,32, value_output]
v_layers = [layer_dims, [], 0, [torch.nn.Softplus(), torch.nn.Softplus(), torch.nn.Softplus(), None]]
value_net = ValueFunction(d_params, LayerInfo(*v_layers), False, 1).to(device)


def dvdx(x, value_net, params):
    value = value_net(x)
    dvdx = torch.autograd.grad(
        value, x, grad_outputs=torch.ones_like(value), create_graph=True, only_inputs=True
    )[0].requires_grad_().view(params.n_batch, 1, params.n_state)
    return dvdx


def project(x, xd, dvdx, loss):
    xd_new = xd - dvdx * (Func.relu((dvdx*xd).sum(dim=1) + loss(x))/(dvdx**2).sum(dim=1))[:, None]
    return xd_new
