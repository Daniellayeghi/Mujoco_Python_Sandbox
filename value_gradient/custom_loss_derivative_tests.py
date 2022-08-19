import numpy as np
from mujoco.derivative import *
from utilities.mj_utils import MjBatchOps
from utilities.torch_utils import *
from collections import namedtuple
import mujoco
import torch
from inv_value import  ValueFunction
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from mlp_torch import MLP
from net_utils_torch import LayerInfo
from loss_functions import *
import loss_functions
torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


m = mujoco.MjModel.from_xml_path(
    "/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml"
)

parent_path = "../../OptimisationBasedControl/data/"
data = pd.read_csv(parent_path + "di_data_value.csv", sep=',', header=None).to_numpy()
n_traj, n_train = 5000, int(5000 * 0.75)

d_train = to_variable(torch.Tensor(data[0:n_train, :]), torch.cuda.is_available()).requires_grad_()
d_test = data[n_train:, :]


batch_size = 1
DataParams = namedtuple('DataParams', 'n_full_state, n_state, n_pos, n_vel, n_ctrl, n_desc, idx_g_act, n_batch')
d_params = DataParams(3, 2, 1, 1, 1, 2, [1, 2], batch_size)
d = mujoco.MjData(m)
bo = MjBatchOps(m, d_params)

d_train_d = TensorDataset(d_train)
d_loader = DataLoader(d_train_d, batch_size=d_params.n_batch, shuffle=False)

# Value network
val_input, value_output = d_params.n_state + d_params.n_desc, d_params.n_ctrl
v_layers = [[val_input, 16, value_output], [], 0]
value_net = ValueFunction(d_params, LayerInfo(*v_layers)).to(device)


x_full = torch.randn(batch_size, d_params.n_state + d_params.n_desc, dtype=torch.double).requires_grad_() * 0.01
x_full[0, :] = torch.tensor([0.5442, -1.03142738, -0.02782536, 1, 2])
x_full_np = tensor_to_np(x_full)
u_star = torch.zeros(batch_size, d_params.n_state + d_params.n_vel, dtype=torch.double)

if __name__ == "__main__":
    loss_functions.set_value_net__(value_net)
    loss_functions.set_batch_ops__(bo)
    # torch.autograd.gradcheck(ctrl_effort_loss.apply, (x_full), rtol=1e-1, atol=1e-1)
    # torch.autograd.gradcheck(ctrl_clone_loss.apply, (x_full, u_star), rtol=1e-1, atol=1e-1)
    torch.autograd.gradcheck(value_lie_loss.apply, (x_full, u_star))
