import mujoco
from mujoco import MjModel, MjData
from mujoco import derivative
import torch
import numpy as np


class MjBatchOps:
    def __init__(self, m, params):
        self.data = mujoco.MjData(m)
        self.model = m
        self.params = params

    def _mj_copy_data(self, m: MjModel, d_src: MjData, d_target: MjData):
        d_target.qpos = d_src.qpos
        d_target.qvel = d_src.qvel
        d_target.qacc = d_src.qacc
        d_target.qfrc_applied = d_src.qfrc_applied
        d_target.xfrc_applied = d_src.xfrc_applied
        d_target.ctrl = d_src.ctrl
        mujoco.mj_forward(m, d_target)

    def f_fwd(self, pos: torch.Tensor, vel: torch.Tensor, u: torch.Tensor, d: MjData, m: MjModel):
        d.qpos = pos
        d.qvel = vel
        d.ctrl = u
        mujoco.mj_step(m, d)
        return torch.Tensor(np.hstack((d.vel, d.qacc)))

    def f_inv(self, pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor, d: MjData, m: MjModel):
        d.qpos, d.qvel, d.qacc = pos, vel, acc
        mujoco.mj_inverse(m, d)
        return torch.Tensor(d.qfrc_inverse)

    def df_inv_dx(self, pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor, d: derivative.MjDataVecView,
                  deriv: derivative.MjDerivative):
        d.qpos, d.qvel, d.qacc = pos, vel, acc
        deriv.func(d)

    def df_fwd_dx(self, pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor, deriv: derivative.MjDerivative):
        self.data.qpos, self.data.qvel, self.data.qacc = pos.numpy().astype('float64'), vel.numpy().astype('float64'), acc.numpy().astype('float64')
        deriv.func(self.data)

    def batch_f_inv(self, frc_applied: torch.Tensor, pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor, d: MjData):
        for i in range(pos.size()[0]):
            self._mj_copy_data(self.model, d, self.data)
            mujoco.mj_inverse(self.model, self.data)
            frc_applied[i, :] = self.f_inv(pos[i, :], vel[i, :], acc[i, :], self.data, self.model)

    def batch_f_inv2(self, frc_applied: torch.Tensor, x: torch.Tensor, d: MjData, params):
        for i in range(x.size()[0]):
            self._mj_copy_data(self.model, d, self.data)
            mujoco.mj_inverse(self.model, self.data)
            pos, vel, acc = x[i, :params.n_pos], x[i, params.n_pos:params.n_vel], x[i, params.n_vel:]
            frc_applied[i, :] = self.f_inv(pos, vel, acc, self.data, self.model)

    def batch_f(self, pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor, u: torch.Tensor, d: MjData):
        for i in range(pos.size()[0]):
            self._mj_copy_data(self.model, d, self.data)
            mujoco.mj_forward(self.model, self.data)
            acc[i, :] = self.f_fwd(pos[i, :], vel[i, :], u[i, :], self.data, self.model)

    def batch_f2(self, x_d: torch.Tensor, x: torch.Tensor, u: torch.Tensor, d: MjData):
        for i in range(x.size()[0]):
            self._mj_copy_data(self.model, d, self.data)
            mujoco.mj_forward(self.model, self.data)
            pos, vel, ctrl = x[i, :self.params.n_pos], x[i, self.params.n_pos:self.params.n_vel], u[i, :]
            x_d[i, :] = self.f_fwd(pos, vel, ctrl, self.data, self.model)

    def batch_df_inv(self, x: torch.Tensor, d: MjData, deriv: derivative.MjDerivative):
        for i in range(x.size()[0]):
            pos, vel, acc = x[i, :self.params.n_pos], x[i, self.params.n_pos:self.params.n_vel], x[i, self.params.n_state:]
            self.df_inv_dx(pos[i, :], vel[i, :], acc[i, :], d, deriv)

    def batch_df_fwd(self, x: torch.Tensor, deriv: derivative.MjDerivative):
        for i in range(x.size()[1]):
            pos, vel, acc = x[:self.params.n_pos, i], x[self.params.n_pos:self.params.n_state, i], x[self.params.n_state:, i]
            self.df_fwd_dx(pos, vel, acc, deriv)
