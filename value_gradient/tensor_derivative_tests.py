import numpy as np
from scipy.optimize import approx_fprime
from utilities.mj_utils import MjBatchOps
from collections import namedtuple
from utilities.torch_utils import *
from mujoco import MjData
import mujoco
import torch
from utilities.mujoco_torch import torch_mj_inv, torch_mj_set_attributes

m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml")

batch_size = 1
DataParams = namedtuple('DataParams', 'n_full_state, n_state, n_pos, n_vel, n_ctrl, n_desc, idx_g_act, n_batch')
d_params = DataParams(3, 2, 1, 1, 1, 2, [1, 2], batch_size)
d = mujoco.MjData(m)
bo = MjBatchOps(m, d_params)

x_full = torch.randn(batch_size, d_params.n_state + d_params.n_vel, dtype=torch.double).requires_grad_() * 0.6
x_full[0, :] = torch.tensor([0.5442, -1.03142738, -0.02782536])
u_star = torch.zeros(batch_size, d_params.n_state + d_params.n_vel, dtype=torch.double)
u_star_loss = np.ones_like(d.qfrc_inverse)

torch_mj_set_attributes(m, batch_size)


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


def clone_loss(xf):
    set_xfull(d, xf)
    mujoco.mj_inverse(m, d)
    return np.sum(np.square(u_star_loss - d.qfrc_inverse))


t_mj_inv = torch_mj_inv.apply

if __name__ == "__main__":
    x_full_np = tensor_to_np(x_full.flatten())
    J_inv = approx_fprime(x_full_np, f_inv, 1e-6)
    J_effort = approx_fprime(x_full_np, effort_loss, 1e-6)
    J_clone = approx_fprime(x_full_np, clone_loss, 1e-6)

    set_xfull(d, x_full_np)
    mujoco.mj_inverse(m, d)
    print(f"Finv mujoco: \n{d.qfrc_inverse}")
    qfrc_torch = t_mj_inv(x_full)
    print(f"Finv torch: \n{qfrc_torch}")

    x_full.retain_grad()
    qfrc_torch.retain_grad()
    qfrc_torch.backward()

    print(f"dFinvdx torch deriv:\n{x_full.grad}")
    print(f"dFinvdx Scipy deriv:\n{J_inv}")
    print(f"dFinvdx FD deriv:\n{bo.b_dfinvdx_full(x_full)}")

    print(f"Effort Scipy deriv:\n{J_effort}")
    print(f"Effort FD deriv:\n{2 * bo.b_qfrcs(x_full) * bo.b_dfinvdx_full(x_full)}")

    print(f"Effort Scipy deriv:\n{J_clone}")
    print(f"Effort FD deriv:\n{2 * (bo.b_qfrcs(x_full).detach().numpy() - u_star_loss) * bo.b_dfinvdx_full(x_full).detach().numpy()}")
