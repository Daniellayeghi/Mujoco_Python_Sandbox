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
DataParams = namedtuple('DataParams', 'n_full_state, n_state, n_pos, n_vel, n_ctrl, n_desc, idx_g_act, n_batch')
d_params = DataParams(3, 2, 1, 1, 1, 2, [1, 2], batch_size)
d = mujoco.MjData(m)
bo = MjBatchOps(m, d_params)
x_full = torch.randn(batch_size, d_params.n_state + d_params.n_vel, dtype=torch.double).requires_grad_() * 0.01
x_full[0, :] = torch.tensor([0.5442, -1.03142738, -0.02782536])
x_full_np = tensor_to_np(x_full)
u_star = torch.zeros(batch_size, d_params.n_state + d_params.n_vel, dtype=torch.double)


class finv(Function):
    @staticmethod
    def forward(ctx, x_full):
        x_full_cpu = x_full.cpu()
        x_full_np = tensor_to_np(x_full)
        qfrcs = bo.b_qfrcs(x_full_np)
        dfinvdx = bo.b_dfinvdx_full(x_full_np)
        ctx.save_for_backward(np_to_tensor(dfinvdx))
        return np_to_tensor(qfrcs)

    @staticmethod
    def backward(ctx, grad_output):
        dfinvdx = ctx.saved_tensors[0]
        return grad_output * dfinvdx


if __name__ == "__main__":
    loss_functions.set_batch_ops__(bo)

    torch.autograd.gradcheck(ctrl_effort_loss.apply, (x_full), rtol=1e-1, atol=1e-1)
    torch.autograd.gradcheck(ctrl_clone_loss.apply, (x_full, u_star), rtol=1e-1, atol=1e-1)
