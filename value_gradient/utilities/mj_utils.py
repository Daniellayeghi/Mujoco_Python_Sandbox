
import mujoco
from mujoco import MjModel, MjData
from mujoco import derivative
import torch
import numpy as np
from typing import List


def mj_copy_data(m: MjModel, d_src: MjData, d_target: MjData):
    d_target.qpos = d_src.qpos
    d_target.qvel = d_src.qvel
    d_target.qacc = d_src.qacc
    d_target.qfrc_applied = d_src.qfrc_applied
    d_target.xfrc_applied = d_src.xfrc_applied
    d_target.ctrl = d_src.ctrl
    mujoco.mj_forward(m, d_target)


def f(pos: torch.Tensor, vel: torch.Tensor, u: torch.Tensor, d: MjData, m: MjModel):
    d.qpos = pos
    d.qvel = vel
    d.ctrl = u
    mujoco.mj_step(m, d)
    return torch.Tensor(np.hstack((d.vel, d.qacc)))


def f_inv(pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor, d: MjData, m: MjModel):
    d.qpos, d.qvel, d.qacc = pos, vel, acc
    mujoco.mj_inverse(m, d)
    return torch.Tensor(d.qfrc_inverse)


def batch_f_inv(frc_applied: torch.Tensor, pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor, d: MjData, d_cp: MjData, m: MjModel):
    for i in range(pos.size()[0]):
        mj_copy_data(m, d, d_cp)
        mujoco.mj_inverse(m, d_cp)
        frc_applied[i, :] = f_inv(pos[i, :], vel[i, :], acc[i, :], d_cp, m)


def batch_f_inv2(frc_applied: torch.Tensor, x: torch.Tensor, d: MjData, d_cp: MjData, m: MjModel, params):
    for i in range(x.size()[0]):
        mj_copy_data(m, d, d_cp)
        mujoco.mj_inverse(m, d_cp)
        pos, vel, acc = x[i, :params.n_pos], x[i, params.n_pos:params.n_vel], x[i, params.n_vel:]
        frc_applied[i, :] = f_inv(pos, vel, acc, d_cp, m)


def batch_f(pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor, u: torch.Tensor, d: MjData, d_cp: MjData, m: MjModel):
    for i in range(pos.size()[0]):
        mj_copy_data(m, d, d_cp)
        mujoco.mj_forward(m, d_cp)
        acc[i, :] = f(pos[i, :], vel[i, :], u[i, :], d_cp, m)


def batch_f2(x_d: torch.Tensor, x: torch.Tensor, u: torch.Tensor, d: MjData, d_cp: MjData, m: MjModel, params):
    for i in range(x.size()[0]):
        mj_copy_data(m, d, d_cp)
        mujoco.mj_forward(m, d_cp)
        pos, vel, ctrl = x[i, :params.n_pos], x[i, params.n_pos:params.n_vel], u[i, :]
        x_d[i, :] = f(pos, vel, ctrl, d_cp, m)


def df_inv_dx(pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor, d: derivative.MjDataVecView, dx: List[derivative.MjDerivative], params):
        d.qpos, d.qvel, d.qacc = pos, vel, acc
        for d in dx:
            d.wrt_dynamics(d)


def df_dx(pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor, d: derivative.MjDataVecView, dx: List[derivative.MjDerivative], params):
    d.qpos, d.qvel, d.qacc = pos, vel, acc
    for d in dx:
        d.wrt_dynamics(d)
