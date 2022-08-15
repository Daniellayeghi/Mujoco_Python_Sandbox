import mujoco
from mujoco import MjModel, MjData
from mujoco import derivative
import torch
import numpy as np
from mujoco.derivative import *


class MjBatchOps:
    def __init__(self, m = None, params = None):
        if m is not None:
            self.__model = m
            self.__data = mujoco.MjData(m)
            self.__dfdx = MjDerivative(m, self.__data, MjDerivativeParams(1e-6, Wrt.State, Mode.Fwd))
            self.__dfdu = MjDerivative(m, self.__data, MjDerivativeParams(1e-6, Wrt.Ctrl, Mode.Fwd))
            self.__dfinvdx = MjDerivative(m, self.__data, MjDerivativeParams(1e-6, Wrt.State, Mode.Inv))
            self.params = params

    def _mj_copy_data(self, m: MjModel, d_src: MjData, d_target: MjData):
        d_target.qpos = d_src.qpos
        d_target.qvel = d_src.qvel
        d_target.qacc = d_src.qacc
        d_target.qfrc_applied = d_src.qfrc_applied
        d_target.xfrc_applied = d_src.xfrc_applied
        d_target.ctrl = d_src.ctrl
        mujoco.mj_forward(m, d_target)

    def _mj_set_ctrl(self, u):
        self.__data.ctrl = u

    def _mj_set_x_full_decomp(self, pos, vel, acc):
        self.__data.qpos, self.__data.qvel, self.__data.qacc = pos, vel, acc

    def _mj_set_state(self, pos, vel):
        self.__data.qpos, self.__data.qvel = pos, vel

    def _mj_set_x_full(self, x):
        self._mj_set_x_full_decomp(
            x[:self.params.n_pos], x[self.params.n_pos:self.params.n_state], x[self.params.n_state:]
        )

    def _mj_set_x(self, x):
        self._mj_set_state(x[:self.params.n_pos], x[self.params.n_pos:self.params.n_state])

    def _mj_set_x_decomp_ctrl(self, pos, vel, u):
        self._mj_set_ctrl(u)
        self._mj_set_state(pos, vel)

    def _mj_set_x_ctrl(self, x, u):
        self._mj_set_ctrl(u)
        self._mj_set_state(x[:self.params.n_pos], x[self.params.n_pos:self.params.n_state])

    def f(self, tensor=True):
        mujoco.mj_step(self.__model, self.__data)
        res = np.hstack((self.__data.qvel, self.__data.qacc)).flatten()
        if tensor:
            return torch.Tensor(res)
        else:
            return res

    def finv(self, tensor=True):
        mujoco.mj_inverse(self.__model, self.__data)
        res = self.__data.qfrc_inverse.flatten()
        if tensor:
            return torch.Tensor(res)
        else:
            return res

    def b_finv_x_full_decomp(self, frc_applied, pos, vel, acc):
        tensor = type(frc_applied) is torch.Tensor
        for i in range(pos.shape[0]):
            self._mj_set_x_full_decomp(pos[i, :], vel[i, :], acc[i, :])
            frc_applied[i, :] = self.finv(tensor)

    def b_finv_x_full(self, frc_applied, x_full):
        tensor = type(frc_applied) is torch.Tensor
        for i in range(x_full.shape[0]):
            self._mj_set_x_full(x_full[i, :])
            frc_applied[i, :] = self.finv(tensor)

    def b_f_x_decomp(self, x_d, pos, vel, u):
        tensor = type(x_d) is torch.Tensor
        for i in range(pos.shape[0]):
            self._mj_set_x_decomp_ctrl(pos[i, :], vel[i, :], u[i, :])
            x_d[i, :] = self.f(tensor)

    def b_f_x(self, x_d, x, u):
        tensor = type(x_d) is torch.Tensor
        for i in range(x.shape[0]):
            self._mj_set_x_ctrl(x[i, :], u[i, :])
            x_d[i, :] = self.f(tensor)

    def b_dfinvdx_full(self, res, x_full):
        tensor = type(res) is torch.Tensor
        for i in range(x_full.shape[0]):
            self._mj_set_x_full(x_full[i, :])
            if tensor:
                res[i, :] = torch.Tensor(self.__dfinvdx.func().flatten())
            else:
                res[i, :] = self.__dfinvdx.func().flatten()

    def b_dfinvdx_full_decomp(self, res, pos, vel, acc):
        tensor = type(res) is torch.Tensor
        for i in range(pos.shape[0]):
            self._mj_set_x_full_decomp(pos[i, :], vel[i, :], acc[i, :])
            if tensor:
                res[i, :] = torch.Tensor(self.__dfinvdx.func().flatten())
            else:
                res[i, :] = self.__dfinvdx.func().flatten()

    def b_dfdx(self, res, x, u):
        tensor = type(res) is torch.Tensor
        for i in range(x.shape[0]):
            self._mj_set_x_ctrl(x[i, :], u[i, :])
            if tensor:
                res[i, :] = torch.Tensor(self.__dfdx.func().flatten())
            else:
                res[i, :] = self.__dfdx.func().flatten()

    def b_dfdu(self, res, x, u):
        tensor = type(res) is torch.Tensor
        for i in range(x.shape[0]):
            self._mj_set_x_ctrl(x[i, :], u[i, :])
            if tensor:
                res[i, :] = torch.Tensor(self.__dfdu.func().flatten())
            else:
                res = self.__dfdu.func().flatten()

    def b_dfdx_decomp(self, res, pos, vel, u):
        tensor = type(res) is torch.Tensor
        for i in range(pos.shape[0]):
            self._mj_set_x_decomp_ctrl(pos[i, :], vel[i, :], u[i, :])
            if tensor:
                res[i, :] = torch.Tensor(self.__dfdx.func().flatten())
            else:
                res = self.__dfdx.func().flatten()
