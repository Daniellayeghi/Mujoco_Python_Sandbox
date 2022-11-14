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
from utilities.mujoco_torch import torch_mj_inv, torch_mj_set_attributes, torch_mj_detach, torch_mj_attach
import mujoco


class PointMassData:
    def __init__(self, d_info: DataParams):
        self.q = torch.rand(d_info.n_batch, 1, n_ees).to(device)
        self.qd = torch.zeros(d_info.n_batch, 1, n_ees).to(device)
        self.qdd = torch.zeros(d_info.n_batch, 1, n_ees).to(device)
        self.d_info = d_info

    def get_x(self):
        return torch.cat((self.q, self.qd), 2).view(d_info.n_batch, 1, self.d_info.n_state).clone()

    def get_xd(self):
        return torch.cat((self.qd, self.qdd), 2).view(d_info.n_batch, 1, self.d_info.n_state).clone()

    def get_xxd(self):
        return torch.cat((self.q, self.qd, self.qdd), 2).view(d_info.n_batch, 1, self.d_info.n_full_state).clone()

    def detach(self):
        self.q.detach()
        self.qd.detach()
        self.qdd.detach()

    def attach(self):
        self.q.requires_grad_()
        self.qd.requires_grad_()
        self.qdd.requires_grad_()


def step_internal(data: PointMassData, dt):
    q_next = data.q + data.qd * dt
    qd_next = data.qd + data.qdd * dt
    data.q = q_next
    data.qd = qd_next


# Data params
n_ees, n_sims = 1, 1
inits = torch.rand(n_ees) * 2
d_info = DataParams(
    n_full_state=3 * n_ees,
    n_state=2 * n_ees,
    n_pos=1 * n_ees,
    n_vel=1 * n_ees,
    n_ctrl=1 * n_ees,
    n_desc=2,
    idx_g_act=[1, 2],
    n_batch=n_sims
)

# Networks and optimizers
val_input, value_output = d_info.n_state, 1
layer_dims = [val_input, 32, 64, 32, value_output]
v_layers = [layer_dims, [], 0, [torch.nn.Softplus(), torch.nn.Softplus(), torch.nn.Softplus(), None]]
value_net = MLP(LayerInfo(*v_layers), False).to(device)

# Mujoco Data
m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml")
x_xd_external = torch.zeros((n_sims, n_ees * d_info.n_full_state, 1)).to(device)


def dvdx(x, value_net):
    value = value_net(x)
    dvdx = torch.autograd.grad(
        value, x, grad_outputs=torch.ones_like(value), create_graph=True, only_inputs=True
    )[0]
    return dvdx


def project(x, xd, x_xd, dvdx, loss):
    norm = (dvdx**2).sum(dim=2).view(n_sims, 1, 1)
    unnorm_porj = Func.relu((dvdx@xd.mT) + loss(x, x_xd))
    xd_trans = - (dvdx/norm) * unnorm_porj
    return xd_trans


def loss(x, x_xd):
    n_sims = len(x)
    loss_task = torch.sum(torch.square(x), 2).view(n_sims, 1, 1)
    # loss_ctrl = torch.sum(torch.square(torch_mj_inverse_func(x_xd)), 1).view(n_sims, 1, 1)
    loss = loss_task #+ loss_ctrl
    return loss


def loss_concat(x_xd):
    n_steps = len(x_xd)
    x = x_xd[:, :, :d_info.n_state].clone()
    x_xd = x_xd.view(n_steps, 1, d_info.n_full_state)
    loss_task = torch.sum(torch.square(x), 2).view(n_steps, 1, 1)
    # loss_ctrl = torch.sum(torch.square(torch_mj_inverse_func(x_xd)), 1).view(n_steps, 1, 1)
    loss = loss_task #+ zloss_ctrl
    return loss


time, epochs, running_loss = 10, 100, 0
buffer = torch.zeros((1, 1, 3)).to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, value_net.parameters()), lr=3e-4)
torch_mj_set_attributes(m, time, n_sims)
torch_mj_inverse_func = torch_mj_inv.apply

if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(epochs):
            d_pm = PointMassData(d_info)
            d = mujoco.MjData(m)
            d.qpos = d_pm.q.cpu().detach().numpy().flatten()
            mass = d.qM
            d_pm.attach()
            torch_mj_attach()
            for t in range(time):
                Vx = dvdx(d_pm.get_x(), value_net)
                xd_trans = project(
                    d_pm.get_x(), d_pm.get_xd(), d_pm.get_xxd(), Vx, lambda x, x_xd: 0.001 * loss(x, x_xd)
                )
                qdd_clone = xd_trans[:, :, 1].clone().view(n_sims, 1, d_info.n_ctrl)
                d_pm.qdd = qdd_clone
                step_internal(d_pm, 0.01)
                buffer = torch.cat((buffer, d_pm.get_xxd()), 0)

            buffer_c = buffer.clone()
            buffer_ds = TensorDataset(buffer_c)
            buffer_loader = DataLoader(buffer_ds, batch_size=time, drop_last=True)

            batch_loss = lambda x_xd: torch.mean(loss_concat(x_xd))
            for i, d in enumerate(buffer_loader):
                d = d[0]
                # optimizer.zero_grad()
                l = batch_loss(d)
                l.backward(retain_graph=True)
                optimizer.step()
                running_loss += l.item()
                avg_loss = running_loss / 20
                print(f"batch: {epoch}, loss: {avg_loss}")

            d_pm.detach()
            torch_mj_detach()
