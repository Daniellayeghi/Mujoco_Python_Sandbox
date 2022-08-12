import mujoco
from mujoco import MjModel, MjData
from mujoco import derivative
import torch
import numpy as np
from mujoco.derivative import *


class MjBatchOps:
    def __init__(self, m, params):
        self.model = m
        self.data = mujoco.MjData(m)
        self.dfdx = MjDerivative(m, self.data, MjDerivativeParams(1e-6, Wrt.State, Mode.Fwd))
        self.dfdu = MjDerivative(m, self.data, MjDerivativeParams(1e-6, Wrt.Ctrl, Mode.Fwd))
        self.dfinvdx = MjDerivative(m, self.data, MjDerivativeParams(1e-6, Wrt.State, Mode.Inv))
        self.params = params

    def _mj_copy_data(self, m: MjModel, d_src: MjData, d_target: MjData):
        d_target.qpos = d_src.qpos
        d_target.qvel = d_src.qvel
        d_target.qacc = d_src.qacc
        d_target.qfrc_applied = d_src.qfrc_applied
        d_target.xfrc_applied = d_src.xfrc_applied
        d_target.ctrl = d_src.ctrl
        mujoco.mj_forward(m, d_target)

    def _mj_set_ctrl(self, u: torch.Tensor):
        self.data.ctrl = u

    def _mj_set_full_x_decomp(self, pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor):
        self.data.qpos, self.data.qvel, self.data.qacc = pos, vel, acc

    def _mj_set_state(self, pos: torch.Tensor, vel: torch.Tensor):
        self.data.qpos, self.data.qvel = pos, vel

    def _mj_set_full_x(self, x: torch.Tensor):
        self._mj_set_full_x_decomp(
            x[:self.params.n_pos], x[self.params.n_pos:self.params.n_state], x[self.params.n_state:]
        )

    def _mj_set_x(self, x: torch.Tensor):
        self._mj_set_state(x[:self.params.n_pos], x[self.params.n_pos:self.params.n_state])

    def _mj_set_x_decomp_ctrl(self, pos: torch.Tensor, vel: torch.Tensor, u: torch.Tensor):
        self._mj_set_ctrl(u)
        self._mj_set_state(pos, vel)

    def _mj_set_x_ctrl(self, x: torch.Tensor, u: torch.Tensor):
        self._mj_set_ctrl(u)
        self._mj_set_state(x[:self.params.n_pos], x[self.params.n_pos:self.params.n_state])

    def f(self):
        mujoco.mj_step(self.model, self.data)
        return torch.Tensor(np.hstack((self.data.qvel, self.data.qacc)).flatten())

    def finv(self):
        mujoco.mj_inverse(self.model, self.data)
        return torch.Tensor(self.data.qfrc_inverse.flatten())

    def b_finv_full_x_decomp(self, frc_applied: torch.Tensor, pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor):
        for i in range(pos.size()[0]):
            self._mj_set_full_x_decomp(pos[i, :], vel[i, :], acc[i, :])
            frc_applied[i, :] = self.finv()

    def b_finv_full_x(self, frc_applied: torch.Tensor, full_x: torch.Tensor):
        for i in range(full_x.size()[0]):
            self._mj_set_full_x(full_x[i, :])
            frc_applied[i, :] = self.finv()

    def b_f_x_decomp(self, x_d: torch.Tensor, pos: torch.Tensor, vel: torch.Tensor, u: torch.Tensor):
        for i in range(pos.size()[0]):
            self._mj_set_x_decomp_ctrl(pos[i, :], vel[i, :], u[i, :])
            x_d[i, :] = self.f()

    def b_f_x(self, x_d: torch.Tensor, x: torch.Tensor, u: torch.Tensor):
        for i in range(x.size()[0]):
            self._mj_set_x_ctrl(x[i, :], u[i, :])
            x_d[i, :] = self.f()

    def b_dfinvdx_full(self, res: torch.Tensor, x_full: torch.Tensor):
        for i in range(x_full.size()[0]):
            self._mj_set_full_x(x_full[i, :])
            res[i, :] = torch.Tensor(self.dfinvdx.func().flatten())

    def b_dfinvdx_full_decomp(self, res: torch.Tensor, pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor):
        for i in range(pos.size()[0]):
            self._mj_set_full_x_decomp(pos[i, :], vel[i, :], acc[i, :])
            res[i, :] = torch.Tensor(self.dfinvdx.func().flatten())

    def b_dfdx(self, res: torch.Tensor, x: torch.Tensor, u: torch.Tensor):
        for i in range(x.size()[0]):
            self._mj_set_x_ctrl(x[i, :], u[i, :])
            res[i, :] = torch.Tensor(self.dfdx.func().flatten())

    def b_dfdu(self, res: torch.Tensor, x: torch.Tensor, u: torch.Tensor):
        for i in range(x.size()[0]):
            self._mj_set_x_ctrl(x[i, :], u[i, :])
            res[i, :] = torch.Tensor(self.dfdu.func().flatten())

    def b_dfdx_decomp(self, res: torch.Tensor, pos: torch.Tensor, vel: torch.Tensor, u: torch.Tensor):
        for i in range(pos.size()[0]):
            self._mj_set_x_decomp_ctrl(pos[i, :], vel[i, :], u[i, :])
            res[i, :] = torch.Tensor(self.dfdx.func().flatten())
