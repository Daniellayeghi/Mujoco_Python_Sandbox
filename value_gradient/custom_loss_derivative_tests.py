import numpy as np
from mujoco.derivative import *
from utilities.mj_utils import MjBatchOps
from utilities.torch_utils import *
from collections import namedtuple
import mujoco
import torch
from loss_functions import *
import loss_functions
torch.autograd.set_detect_anomaly(True)

m = mujoco.MjModel.from_xml_path(
    "/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml"
)

batch_size = 1
DataParams = namedtuple('DataParams', 'n_full_state, n_state, n_pos, n_vel, n_ctrl, n_desc, n_batch')
d_params = DataParams(3, 2, 1, 1, 1, 2, batch_size)
d = mujoco.MjData(m)
bo = MjBatchOps(m, d_params)
x_full      = torch.zeros(batch_size, d_params.n_state + d_params.n_vel, dtype=torch.double)
res_qfrc    = torch.zeros(batch_size, d_params.n_vel, dtype=torch.double).requires_grad_()
res_dfinvdx = torch.zeros(
    batch_size, (d_params.n_state + d_params.n_vel) * d_params.n_vel, dtype=torch.double
).requires_grad_()

if __name__ == "__main__":
    loss_functions.set_batch_ops__(bo)
    torch.autograd.gradcheck(ctrl_effort_loss.apply, (x_full, res_qfrc, res_dfinvdx))


