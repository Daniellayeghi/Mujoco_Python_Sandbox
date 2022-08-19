import numpy as np
from scipy.optimize import approx_fprime
from utilities.mj_utils import MjBatchOps
from collections import namedtuple
from utilities.torch_utils import *
from mujoco import MjData
import mujoco
import torch
import cProfile

m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml")

batch_size = 1
DataParams = namedtuple('DataParams', 'n_full_state, n_state, n_pos, n_vel, n_ctrl, n_desc, idx_g_act, n_batch')
d_params = DataParams(3, 2, 1, 1, 1, 2, [1, 2], batch_size)
d = mujoco.MjData(m)
bo = MjBatchOps(m, d_params)

x_full = torch.randn(batch_size, d_params.n_state + d_params.n_vel, dtype=torch.double).requires_grad_() * 0.01
x_full[0, :] = torch.tensor([0.5442, -1.03142738, -0.02782536])
u_star = torch.zeros(batch_size, d_params.n_state + d_params.n_vel, dtype=torch.double)


def set_xfull(d: MjData, xf):
    d.qpos = xf[:d_params.n_pos]
    d.qvel = xf[d_params.n_pos:d_params.n_state]
    d.qacc = xf[d_params.n_state:]


def f_inv(xf):
    set_xfull(d, xf)
    mujoco.mj_inverse(m, d)
    return d.qfrc_inverse


def effort_loss(xf):
    set_xfull(d, xf)
    mujoco.mj_inverse(m, d)
    return np.sum(np.square(d.qfrc_inverse))


if __name__ == "__main__":
    x_full_np = tensor_to_np(x_full.flatten())
    J_ctrl = approx_fprime(x_full_np, f_inv, 1e-6)
    J_effort = approx_fprime(x_full_np, effort_loss, 1e-6)

    print(f"Finv Scipy deriv:\n{J_ctrl}, for inv {f_inv(x_full_np)}")
    print(f"Finv FD deriv:\n{bo.b_dfinvdx_full(x_full)}")

    print(f"Effort Scipy deriv:\n{J_effort}, for loss {effort_loss(x_full_np)}")
    print(f"Effort FD deriv:\n{2 * bo.b_qfrcs(x_full) * bo.b_dfinvdx_full(x_full)}")

