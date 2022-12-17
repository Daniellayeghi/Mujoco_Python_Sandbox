import numpy as np
from scipy.optimize import approx_fprime
from utilities.mj_utils import MjBatchOps
from collections import namedtuple
from utilities.torch_utils import *
from mujoco import MjData
import mujoco
import torch
from torch.autograd.functional import jacobian
from utilities.mujoco_torch import torch_mj_inv, torch_mj_set_attributes, SimulationParams

m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator_sparse.xml")

d_params = SimulationParams(6, 4, 2, 2, 1, 2, 1, 1)
d = mujoco.MjData(m)

u_star = torch.zeros(1, d_params.nqva, dtype=torch.double)
u_star_loss = np.ones_like(d.qfrc_inverse)

torch_mj_set_attributes(m, d_params)


def set_xfull(d: MjData, xf):
    d.qpos = xf[:d_params.nq]
    d.qvel = xf[d_params.nq:d_params.nqv]
    d.qacc = xf[d_params.nqv:]


def f_inv(xf):
    set_xfull(d, xf)
    mujoco.mj_inverse(m, d)
    return d.qfrc_inverse[1]


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
    x_full = torch.randn(1, 1, 1, d_params.nqva, dtype=torch.double).requires_grad_() * 0.6
    x_full[0, 0, 0, :] = torch.tensor([-.6, 0, 0, 0, 0, 0])
    x_full_np = tensor_to_np(x_full.flatten())
    J_inv = approx_fprime(x_full_np, f_inv, 1e-6)
    J_effort = approx_fprime(x_full_np, effort_loss, 1e-6)
    J_clone = approx_fprime(x_full_np, clone_loss, 1e-6)
    set_xfull(d, x_full_np)
    mujoco.mj_inverse(m, d)
    dfinv_dx = jacobian(t_mj_inv, x_full)
    qfrc_torch = t_mj_inv(x_full)
    print(f"Finv torch:\n{qfrc_torch}, Finv:\n{d.qfrc_inverse}")
    print(f"Finv derivative scipy:\n{J_inv}, Finv derivative torch:\n{dfinv_dx}")
