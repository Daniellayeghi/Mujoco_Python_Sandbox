import numpy
import torch
from scipy.optimize import approx_fprime
from utilities.data_utils import *
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from net_utils_torch import LayerInfo
from net_loss_functions import *
import net_loss_functions
import mujoco
torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


m = mujoco.MjModel.from_xml_path(
    "/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml"
)

# parent_path = "../../OptimisationBasedControl/data/"
# data = pd.read_csv(parent_path + "di_data_value.csv", sep=',', header=None).to_numpy()
# n_traj, n_train = 5000, int(5000 * 0.75)
#
# d_train = to_variable(torch.Tensor(data[0:n_train, :]), torch.cuda.is_available()).requires_grad_()
# d_test = data[n_train:, :]

batch_size = 1
d_params = DataParams(3, 2, 1, 1, 1, 2, [1, 2], batch_size)

# d_train_d = TensorDataset(d_train)
# d_loader = DataLoader(d_train_d, batch_size=d_params.n_batch, shuffle=False)
d = mujoco.MjData(m)
batch_op = MjBatchOps(m, d_params)


# Value network
val_input, value_output = d_params.n_state + d_params.n_desc, d_params.n_ctrl
v_layers = [[val_input, 16, value_output], [], 0]
value_net = ValueFunction(d_params, LayerInfo(*v_layers)).to(device)

x_full = torch.randn(batch_size, d_params.n_full_state).requires_grad_() * 0.01
x_full[0, :] = torch.tensor([0.5442, -1.03142738, -0.02782536])
x_full_np = tensor_to_np(x_full)
x_desc = torch.randn(batch_size, d_params.n_state + d_params.n_desc).to(device).requires_grad_() * 0.01
x_desc_curr = torch.randn(batch_size, d_params.n_state + d_params.n_desc).to(device) * 0.01
x_desc_np = tensor_to_np(x_desc)
u_star = torch.ones(batch_size, d_params.n_ctrl, dtype=torch.float)

if __name__ == "__main__":
    net_loss_functions.set_value_net__(value_net)
    net_loss_functions.set_batch_ops__(batch_op)
    net_loss_functions.set_dt_(0.01)
    loss_func = value_dt_loss.apply
    value_net.update_dvdx_desc(x_desc)
    loss = torch.mean(loss_func(x_desc, x_desc_curr))
    loss.retain_grad()
    x_desc.retain_grad()
    loss.backward()
    print(f"Loss is {loss}, loss.grad {x_desc.grad}")

    torch.autograd.gradcheck(ctrl_effort_loss.apply, (x_full), rtol=1e-1, atol=1e-1)
    torch.autograd.gradcheck(ctrl_clone_loss.apply, (x_full, u_star), rtol=1e1, atol=1e1)
    value_net.dvdxx(x_desc)
    value_net.update_grads_x(x_desc)
    torch.autograd.gradcheck(value_lie_loss.apply, (x_desc, u_star), rtol=1e-6, atol=1e-6)
