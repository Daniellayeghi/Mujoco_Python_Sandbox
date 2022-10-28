import numpy as np
import torch
from utilities import mujoco_torch
from utilities.torch_utils import tensor_to_np
from utilities.torch_device import device
from utilities.data_utils import DataParams
from networks import ValueFunction, MLP
from net_utils_torch import LayerInfo
import torch.nn.functional as Func
from torch.utils.data import TensorDataset, DataLoader
from utilities.mujoco_torch import torch_mj_inv, torch_mj_set_attributes
import mujoco


class PointMassData:
    def __init__(self, n_bodies, rand=True):
        self.x_xd = torch.zeros((n_bodies, 3)).to(device)
        if rand:
            self.x_xd[:, 0] = torch.rand((n_bodies))

        self.x = self.x_xd[:, 0:2]
        self.xd = self.x_xd[:, 2:]
        self.qpos = self.x_xd[:, 0]
        self.qvel = self.x_xd[:, 1]
        self.qacc = self.x_xd[:, 2]

        self.x_xd.requires_grad_()


def step(data: PointMassData, dt):
    data.qvel = data.qvel + data.qacc * dt
    data.qpos = data.qpos + data.qvel * dt


# Data params
batch_size, inits = 16, torch.rand(20) * 2
d_params = DataParams(3, 2, 1, 1, 1, 2, [1, 2], batch_size)

# Networks and optimizers
val_input, value_output = d_params.n_state, 1
layer_dims = [val_input, 32, 64, 32, value_output]
v_layers = [layer_dims, [], 0, [torch.nn.Softplus(), torch.nn.Softplus(), torch.nn.Softplus(), None]]
value_net = ValueFunction(d_params, LayerInfo(*v_layers), False, 1).to(device).requires_grad_()

# Mujoco Data
m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml")
d_pm = PointMassData(len(inits))
d = mujoco.MjData(m)


def dvdx(x, value_net, params):
    value = value_net(x).requires_grad_()
    dvdx = torch.autograd.grad(
        value, x, grad_outputs=torch.ones_like(value), create_graph=True, only_inputs=True
    )[0].requires_grad_().view(params.n_batch, 1, params.n_state)
    return dvdx


def project(x, xd, dvdx, loss):
    xd_new = xd - dvdx * (Func.relu((dvdx*xd).sum(dim=1) + loss(x))/(dvdx**2).sum(dim=1))[:, None]
    return xd_new


m = d.qM
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, value_net.parameters()), lr=1e-4)


def loss(x_xd):
    x = x_xd[:, 0:d_params.n_state]
    loss_task = torch.sum(torch.square(x))
    loss_ctrl = torch.sum(torch.square(torch_mj_inv(x_xd)))
    loss = loss_task + loss_ctrl
    return torch.mean(loss)


time, epochs, running_loss = 100, 100, 0
buffer = torch.zeros((time * len(inits), 3)).to(device).requires_grad_()
buffer_ds = TensorDataset(buffer)


if __name__ == "__main__":

    for epoch in range(epochs):
        for id, init in enumerate(inits):
            for t in range(time):
                dvdx = dvdx(d_pm.x, value_net, d_params)
                d_pm.qacc = project(d_pm.x, d_pm.xd, dvdx, lambda x_xd: 1e-3 * loss(x_xd))[:, -1]
                step(d_pm, 0.01)
                buffer_ds[time * init, :] = d_pm.x_xd

            buffer_loader = DataLoader(buffer_ds, batch_size=d_params.n_batch, shuffle=True, drop_last=True)
            for i, d in enumerate(buffer_loader):
                l = loss(d)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                running_loss += l.item()

                if i % 20 == 0:
                    loss = running_loss / 20
                    print(f"batch: {epoch}, loss: {loss}")
