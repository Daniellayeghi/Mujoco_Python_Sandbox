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
        self.x = torch.zeros((n_bodies, 2)).to(device)

        if rand:
            self.x[:, 0] = torch.rand((n_bodies, 1))

        self.xd = torch.zeros((n_bodies, 2)).to(device)
        self.pos = self.x[:, 0]
        self.vel = self.x[:, 1]
        self.acc = self.xd[:, 1]


def step_internal(data: PointMassData, xd_hat, dt):
    data.x += xd_hat * dt
    data.xd[:, 1] = xd_hat[:, 1]


def step_external(data: PointMassData, x_xd, dt):
    x_xd[:, 0, 0] = data.pos + data.vel * dt
    x_xd[:, 1, 0] = data.vel + data.acc * dt


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


def project(pm_data, dvdx, loss):
    dvdx_batch = dvdx.view(n_bodies, 1, 2)
    x = pm_data.x .view(n_bodies, 3, 1)
    xd = pm_data.xd.view(n_bodies, 1, 1)
    norm = (dvdx_batch**2).sum(dim=2).view(n_bodies, 1, 1)
    unnorm_porj = Func.relu((dvdx_batch@xd) + loss(x, xd))
    delta_xd = - (dvdx_batch/norm) * unnorm_porj
    return delta_xd


optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, value_net.parameters()), lr=3e-4)
torch_mj_set_attributes(m, n_batch, n_bodies)
torch_mj_inverse_func = torch_mj_inv.apply


def loss(x, xd):
    n_batch = len(x)
    x = x.view(n_batch, 2, 1)
    loss_task = torch.sum(torch.square(x), 1).view(n_batch, 1, 1)
    loss_ctrl = torch.sum(torch.square(torch_mj_inverse_func(xd[:, 1])), 1).view(n_batch, 1, 1)
    loss = loss_task + loss_ctrl
    return loss


time, epochs, running_loss = 100, 100, 0
buffer = torch.zeros((time * n_bodies, 3)).to(device)
buffer = [[None] * 3] * (time * n_bodies)


if __name__ == "__main__":

    for epoch in range(epochs):
        d_pm = PointMassData(n_bodies)
        d = mujoco.MjData(m)
        mass = d.qM
        d.qpos = d_pm.pos.cpu().detach().numpy()
        d_pm.x = d_pm.x.cpu().detach().numpy()
        d_pm.qdd = d_pm.qdd.cpu().detach().numpy()
        x = d_pm.x.detach().requires_grad_()
        qdd = d_pm.qdd.detach().requires_grad_()
        for t in range(time):
            Vx = dvdx(x, value_net)
            delta_xd = project(d_pm, Vx, lambda x, qdd: 1 * loss(x, qdd))
            d_pm.xd -= delta_xd.squeeze()
            step_internal(d_pm, 0.01)
            buffer[t] = [d_pm.x, d_pm.xd]

        buffer_d = buffer
        buffer_ds = TensorDataset(buffer_d)
        buffer_loader = DataLoader(buffer_ds, batch_size=d_params.n_batch, shuffle=True, drop_last=True)
        batch_loss = lambda x_xd: torch.mean(loss(x_xd))

        for i, d in enumerate(buffer_loader):
            d = d[0]
            optimizer.zero_grad()
            l = batch_loss(d)
            l.backward()
            optimizer.step()
            running_loss += l.item()

            if i % 5 == 0:
                avg_loss = running_loss / 20
                print(f"batch: {epoch}, loss: {avg_loss}")

        # buffer.detach()
