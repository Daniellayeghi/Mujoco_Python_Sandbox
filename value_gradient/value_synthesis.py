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
        self.xd = self.x_xd[:, 1:]
        self.qpos = self.x_xd[:, 0]
        self.qvel = self.x_xd[:, 1]
        self.qacc = self.x_xd[:, 2]

        self.x_xd


def step_internal(data: PointMassData, dt):
    data.qpos = data.qpos + data.qvel * dt
    data.qvel = data.qvel + data.qacc * dt


def step_external(data: PointMassData, x_xd, dt):
    x_xd[:, 0, 0] = data.qpos + data.qvel * dt
    x_xd[:, 1, 0] = data.qvel + data.qacc * dt
    x_xd[:, 2, 0] = data.qacc


# Data params
n_bodies, n_batch = 1, 16
inits = torch.rand(n_bodies) * 2
d_params = DataParams(3, 2, 1, 1, 1, 2, [1, 2], n_batch)

# Networks and optimizers
val_input, value_output = d_params.n_state, 1
layer_dims = [val_input, 32, 64, 32, value_output]
v_layers = [layer_dims, [], 0, [torch.nn.Softplus(), torch.nn.Softplus(), torch.nn.Softplus(), None]]
value_net = MLP(LayerInfo(*v_layers), False).to(device)

# Mujoco Data
m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml")
x_xd_external = torch.zeros((n_bodies, 3, 1)).to(device)

def dvdx(x, value_net):
    value = value_net(x).requires_grad_()
    dvdx = torch.autograd.grad(
        value, x, grad_outputs=torch.ones_like(value), create_graph=True, only_inputs=True
    )[0].requires_grad_()
    return dvdx


def project(pm_data, x_xd_next, dvdx, loss):
    dvdx_batch = dvdx.view(n_bodies, 1, 2)
    x_xd_batch = pm_data.x_xd.view(n_bodies, 3, 1)
    xd_next = x_xd_next[:, 1:]
    norm = (dvdx_batch**2).sum(dim=2).view(n_bodies, 1, 1)
    unnorm_porj = Func.relu((dvdx_batch@xd_next) + loss(x_xd_batch))
    delta_xd = - (dvdx_batch/norm) * unnorm_porj
    return delta_xd


optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, value_net.parameters()), lr=3e-4)
torch_mj_set_attributes(m, n_batch, n_bodies)
torch_mj_inverse_func = torch_mj_inv.apply


def loss(x_xd):
    n_batch = len(x_xd)
    x = x_xd[:, 0:d_params.n_state].view(n_batch, 2, 1)
    loss_task = torch.sum(torch.square(x), 1).view(n_batch, 1, 1)
    loss_ctrl = torch.sum(torch.square(torch_mj_inverse_func(x_xd)), 1).view(n_batch, 1, 1)
    loss = loss_task + loss_ctrl
    return loss


time, epochs, running_loss = 100, 100, 0
buffer = torch.zeros((time * n_bodies, 3)).to(device)


if __name__ == "__main__":

    for epoch in range(epochs):
        d_pm = PointMassData(n_bodies)
        d = mujoco.MjData(m)
        d.qpos = d_pm.qpos.cpu().detach().numpy()
        d.qvel = d_pm.qvel.cpu().detach().numpy()
        d.qacc = d_pm.qacc.cpu().detach().numpy()
        mass = d.qM
        for t in range(time):
            x = d_pm.x.detach().requires_grad_()
            Vx = dvdx(x, value_net)
            step_external(d_pm, x_xd_external, 0.01)
            delta_xd = project(d_pm, x_xd_external, Vx, lambda x_xd: 1e-3 * loss(x_xd))[:, -1]
            d_pm.qacc = d_pm.qacc + delta_xd[:, -1]
            step_internal(d_pm, 0.01)
            buffer[t * n_bodies:(t * n_bodies) + n_bodies, :] = d_pm.x_xd

        buffer_d = buffer.detach().requires_grad_()
        buffer_ds = TensorDataset(buffer_d)
        buffer_loader = DataLoader(buffer_ds, batch_size=d_params.n_batch, shuffle=True, drop_last=True)
        batch_loss = lambda x_xd: torch.mean(loss(x_xd))
        for i, d in enumerate(buffer_loader):
            d = d[0].detach().requires_grad_()
            optimizer.zero_grad()
            l = batch_loss(d)
            l.backward()
            optimizer.step()
            running_loss += l.item()

            if i % 5 == 0:
                avg_loss = running_loss / 20
                print(f"batch: {epoch}, loss: {avg_loss}")
