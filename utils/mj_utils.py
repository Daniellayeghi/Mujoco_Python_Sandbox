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


def mj_frc_from_inverse(frc_applied: torch.Tensor, pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor, d: MjData, m: MjModel):
    d.qpos = pos
    d.qvel = vel
    d.qacc = acc
    mujoco.mj_inverse(m, d)
    frc_applied = d.qfrc_applied


def mj_multi_inverse(frc_applied: torch.Tensor, pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor, d, m: MjModel, params):
    d_cp = MjData(m)
    for i in range(pos.size()[0]):
        mj_copy_data(m, d, d_cp)
        beg_frc, end_frc = params.n_ctrl * i, (params.n_ctrl * i) + params.n_ctrl
        beg_state, end_state = params.n_pos * i, (params.n_pos * i) + params.n_pos
        mj_frc_from_inverse(
            frc_applied[beg_frc:end_frc],
            pos[beg_state:end_state],
            vel[beg_state:end_state],
            acc[beg_state:end_state],
            d_cp,
            m
        )


def mj_multi_inverse_par(frc_applied: torch.Tensor, pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor, ds: List[MjData], m: MjModel, params):
    with parallel_backend('threading', n_jobs=5):
        Parallel()(
            delayed(mj_frc_from_inverse)(
                frc_applied[params.n_ctrl * i:(params.n_ctrl * i) + params.n_ctrl],
                pos[params.n_pos * i:(params.n_pos * i) + params.n_pos],
                vel[params.n_pos * i:(params.n_pos * i) + params.n_pos],
                acc[params.n_pos * i:(params.n_pos * i) + params.n_pos],
                ds[i],
                m
            ) for i in range(pos.size()[0]))


if __name__ == "__main__":
    acc = torch.rand(300, 1)
    vel = torch.zeros_like(acc)
    pos = torch.zeros_like(vel)
    frc = torch.zeros_like(pos)

    def integrate(x1, x2, dt):
        x1 += x2 * dt

    integrate(vel, acc, 0.01)
    integrate(pos, vel, 0.01)
    m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator_sparse.xml")
    d = MjData(m)
    ds = [MjData(m) for _ in range(5)]
    mujoco.mj_step(m, d)
    DataParams = namedtuple('DataParams', 'n_state, n_pos, n_vel, n_ctrl')
    d_params = DataParams(2, 1, 1, 1)
    mj_multi_inverse(frc, pos, vel, acc, d, m, d_params)
    mj_multi_inverse_par(frc, pos, vel, acc, ds, m, d_params)
    print("Done!")


