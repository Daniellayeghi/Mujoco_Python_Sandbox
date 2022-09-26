import sys
import numpy as np
from mlp_torch import MLP
from net_utils_torch import LayerInfo
from utilities.data_utils import *
from ioc_loss_di import *
from networks import ValueFunction, OptimalPolicy
from torch_device import device, is_cuda
import mujoco
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from livelossplot import PlotLosses

# Data params
batch_size = 16
d_params = DataParams(3, 2, 1, 1, 1, 0, [1, 2], batch_size)

# Mujoco models
m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml")
d = mujoco.MjData(m)

# Load Data
parent_path = "../../OptimisationBasedControl/data/"
u_data = pd.read_csv(parent_path + "test_di_ctrl3.csv", sep=',', header=None).to_numpy()
x_data = pd.read_csv(parent_path + "test_di_state3.csv", sep=',', header=None).to_numpy()
data = np.hstack((x_data, u_data))
n_traj, n_train = data.shape[0], int(data.shape[0] * 0.75)
d_train = torch.Tensor(data[0:n_train, :]).to(device)
d_test = data[n_train:, :]

d_train_d = TensorDataset(d_train)
d_loader = DataLoader(d_train_d, batch_size=d_params.n_batch, shuffle=True, drop_last=True)

# Networks and optimizers
val_input, value_output = d_params.n_state, 1
layer_dims = [val_input, 16, 16, value_output]
v_layers = [layer_dims, [], 0, [torch.nn.Softplus(), torch.nn.Softplus(), torch.nn.Softplus(), None]]
value_net = ValueFunction(d_params, LayerInfo(*v_layers), False, 1).to(device)


lr = 1e-4
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, value_net.parameters()), lr=lr)
# lr_s = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=- 1)

set_value_net__(value_net)
loss_v = loss_value_proj.apply


b_Rinv = torch.ones((batch_size, 1, 1))
b_Btran = torch.tensor([0.0005, 0.09999975]).repeat(d_params.n_batch, 1, 1)
b_B_R = 0.5 * torch.bmm(b_Rinv, b_Btran).to(device)
goal = torch.zeros((1, d_params.n_state)).to(device)


def b_l2_loss(x, u_star):
    v = value_net(x).requires_grad_()
    dvdx = torch.autograd.grad(
        v, x, grad_outputs=torch.ones_like(v), create_graph=True
    )[0].requires_grad_().view(d_params.n_batch, 1, d_params.n_state)

    b_proj_v = torch.bmm(b_B_R, dvdx.mT)
    b_loss = u_star.view(d_params.n_batch, d_params.n_ctrl, 1) + b_proj_v

    loss = torch.mean(b_loss.square().sum(2))
    return loss


running_loss = 0
try:
    for epoch in range(150):
        logs = {}
        for i, d in enumerate(d_loader):
            # TODO: Maybe this copy is unnecessary
            x = (d[0][:, :-d_params.n_ctrl]).requires_grad_()
            u_star = d[0][:, d_params.n_state:]
            # value_net.update_grads(x)
            # Detach x_desc_curr so its derivatives are ignored
            loss = b_l2_loss(x, u_star)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.detach() * len(d)

        epoch_loss = running_loss / len(d_loader)
        running_loss = 0
        print('loss: {} epoch: {} lr: {}'.format(epoch_loss.item(), epoch, optimizer.param_groups[0]['lr']))
        # lr_s.step()

    stored_exception = sys.exc_info()
    print("########## Saving Trace ##########")
    torch.save(value_net.state_dict(), "./op_value_di6.pt")

except KeyboardInterrupt:
    stored_exception = sys.exc_info()
    print("########## Saving Trace ##########")
    torch.save(value_net.state_dict(), "./op_value_di6.pt")
