from collections import namedtuple

import mujoco
import torch
from mujoco import MjModel, MjData
from typing import List
from joblib import parallel_backend, Parallel, delayed


def mj_copy_data(m: MjModel, d_src: MjData, d_target: MjData):
    d_target.qpos = d_src.qpos
    d_target.qvel = d_src.qvel
    d_target.qacc = d_src.qacc
    d_target.qfrc_applied = d_src.qfrc_applied
    d_target.xfrc_applied = d_src.xfrc_applied
    d_target.ctrl = d_src.ctrl
    mujoco.mj_forward(m, d_target)


def mj_frc_from_inverse(pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor, d: MjData, m: MjModel):
    d.qpos = pos
    d.qvel = vel
    d.qacc = acc
    mujoco.mj_inverse(m, d)
    return torch.Tensor(d.qfrc_inverse)


def mj_batch_inverse(frc_applied: torch.Tensor, pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor, d, m: MjModel, params):
    d_cp = MjData(m)
    for i in range(pos.size()[0]):
        mj_copy_data(m, d, d_cp)
        beg_frc, end_frc = params.n_ctrl * i, (params.n_ctrl * i) + params.n_ctrl
        beg_state, end_state = params.n_pos * i, (params.n_pos * i) + params.n_pos
        frc_applied[beg_frc:end_frc] = mj_frc_from_inverse(
            pos[beg_state:end_state],
            vel[beg_state:end_state],
            acc[beg_state:end_state],
            d_cp,
            m
        )

def mj_batch_derivative()