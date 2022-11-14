import math
import numpy as np

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
from networks import ValueFunction, MLP
from net_utils_torch import LayerInfo
from utilities.mujoco_torch import torch_mj_inv, torch_mj_set_attributes, torch_mj_detach, torch_mj_attach
import mujoco

use_cuda = torch.cuda.is_available()


def euler_step(z0, t0, t1, f):
    """
    Simplest Euler ODE initial value solver
    """
    h_max = 0.01
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())

    h = (t1 - t0)/n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
        z = z + h * f(z, t)
        t = t + h
    return z


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