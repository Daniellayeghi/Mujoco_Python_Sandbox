import numpy as np
import torch
from utilities import  mujoco_torch
from utilities.torch_device import device
import mujoco


class PointMassData:
    def __init__(self, n_bodies):
        self.x_xd = np.zeros((3,n_bodies))
        self.qpos = self.x_xd[0, :]
        self.qvel = self.x_xd[1, :]
        self.qacc = self.x_xd[2, :]


def step(data: PointMassData, dt):
    data.qvel = data.qvel + data.qacc * dt
    data.qpos = data.qpos + data.qvel * dt


# Data params
batch_size = 16
d_params = DataParams(3, 2, 1, 1, 1, 2, [1, 2], batch_size)

# Networks and optimizers
val_input, value_output = d_params.n_state, 1
layer_dims = [val_input, 32, 64,32, value_output]
v_layers = [layer_dims, [], 0, [torch.nn.Softplus(), torch.nn.Softplus(), torch.nn.Softplus(), None]]
value_net = ValueFunction(d_params, LayerInfo(*v_layers), False, 1).to(device)
