import mujoco
from mujoco import MjModel, MjData
from mujoco import derivative
import torch
import numpy as np
from mujoco.derivative import *


class MjBatchOps:
    def __init__(self, m, params):
        self.__model = m
        self.params = params
        self.__data = mujoco.MjData(m)
        self.__dfdx = MjDerivative(m, self.__data, MjDerivativeParams(1e-6, Wrt.State, Mode.Fwd))
        self.__dfdu = MjDerivative(m, self.__data, MjDerivativeParams(1e-6, Wrt.Ctrl, Mode.Fwd))
        self.__dfinvdx = MjDerivative(m, self.__data, MjDerivativeParams(1e-6, Wrt.State, Mode.Inv))

        self.__qfrc_t = torch.zeros(
            (self.params.n_batch, self.params.n_vel), dtype=torch.double
        ).requires_grad_()

        self.__dxdt_t = torch.zeros(
            (self.params.n_batch, self.params.n_vel * 2), dtype=torch.double
        ).requires_grad_()

        self.__dfinvdx_t = torch.zeros(
            (self.params.n_batch, self.params.n_full_state * self.params.n_vel), dtype=torch.double
        ).requires_grad_()

        self.__dfdx_t = torch.zeros(
            (self.params.n_batch, self.params.n_state**2), dtype=torch.double
        ).requires_grad_()

        self.__dfdu_t = torch.zeros(
            (self.params.n_batch, self.params.n_ctrl*self.params.n_state), dtype=torch.double
        ).requires_grad_()

        self.__gu_t = torch.zeros(
            (self.params.n_batch, self.params.n_state, self.params.n_ctrl), dtype=torch.double
        ).requires_grad_()

        self.__u_t = torch.zeros(
            (self.params.n_batch, self.params.n_ctrl), dtype=torch.double
        ).requires_grad_()

        self.__gu_t[:, self.params.g_act_idx[0]:self.params.g_act_idx[1], self.params.n_ctrl] = 1

        self.__qfrc_np = self.__qfrc_t.detach().numpy().astype('float64')
        self.__dfinvdx_np = self.__dfinvdx_t.detach().numpy().astype('float64')
        self.__dxdt_np = self.__dxdt_t.detach().numpy().astype('float64')
        self.__dfdx_np = self.__dfdx_t.detach().numpy().astype('float64')
        self.__dfdu_np = self.__dfdu_t.detach().numpy().astype('float64')
        self.__gu_np = self.__gu_t.detach().numpy().astype('float64')

    @staticmethod
    def _get_result(operation, tensor_flag):
        if tensor_flag:
            return torch.Tensor(operation.func().flatten())
        else:
            return operation.func().flatten()

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

    def b_qfrc_decomp(self, pos, vel, acc):
        tensor = type(pos) is torch.Tensor
        res = self.__qfrc_t if tensor else self.__qfrc_np
        with torch.no_grad():
            for i in range(pos.shape[0]):
                self._mj_set_x_full_decomp(pos[i, :], vel[i, :], acc[i, :])
                res[i, :] = self.finv(tensor)
        return res

    def b_qfrcs(self, x_full):
        tensor = type(x_full) is torch.Tensor
        res = self.__qfrc_t if tensor else self.__qfrc_np
        with torch.no_grad():
            for i in range(x_full.shape[0]):
                self._mj_set_x_full(x_full[i, :])
                res[i, :] = self.finv(tensor)
        return res

    def b_dxdt_decomp(self, pos, vel, u):
        tensor = type(pos) is torch.Tensor
        res = self.__dfdx_t if tensor else self.__dfdx_np
        with torch.no_grad():
            for i in range(pos.shape[0]):
                self._mj_set_x_decomp_ctrl(pos[i, :], vel[i, :], u[i, :])
                res[i, :] = self.f(tensor)
        return res

    def b_dxdt(self, x, u):
        tensor = type(x) is torch.Tensor
        res = self.__dxdt_t if tensor else self.__dxdt_np
        with torch.no_grad():
            for i in range(x.shape[0]):
                self._mj_set_x_ctrl(x[i, :], u[i, :])
                res[i, :] = self.f(tensor)
        return res

    def b_dfinvdx_full(self, x_full):
        tensor = type(x_full) is torch.Tensor
        res = self.__dfinvdx_t if tensor else self.__dfinvdx_np
        with torch.no_grad():
            for i in range(x_full.shape[0]):
                self._mj_set_x_full(x_full[i, :])
                res[i, :] = self._get_result(self.__dfinvdx, tensor)
        return res

    def b_dfinvdx_full_decomp(self, pos, vel, acc):
        tensor = type(pos) is torch.Tensor
        res = self.__dfinvdx_t if tensor else self.__dfinvdx_np
        with torch.no_grad():
            for i in range(pos.shape[0]):
                self._mj_set_x_full_decomp(pos[i, :], vel[i, :], acc[i, :])
                res[i, :] = self._get_result(self.__dfinvdx, tensor)
        return  res

    def b_dfdx(self, x, u):
        tensor = type(x) is torch.Tensor
        with torch.no_grad():
            res = self.__dfdx_t if tensor else self.__dfdx_np
            for i in range(x.shape[0]):
                self._mj_set_x_ctrl(x[i, :], u[i, :])
                res[i, :] = self._get_result(self.__dfdx, tensor)
        return res

    def b_dfdx_decomp(self, pos, vel, u):
        tensor = type(pos) is torch.Tensor
        res = self.__dfdx_t if tensor else self.__dfdx_np
        with torch.no_grad():
            for i in range(pos.shape[0]):
                self._mj_set_x_decomp_ctrl(pos[i, :], vel[i, :], u[i, :])
                res[i, :] = self._get_result(self.__dfdx, tensor)
        return res

    def b_dfdu(self, x, u):
        tensor = type(x) is torch.Tensor
        res = self.__dfdu_t if tensor else self.__dfdu_np
        with torch.no_grad():
            for i in range(x.shape[0]):
                self._mj_set_x_ctrl(x[i, :], u[i, :])
                res[i, :] = self._get_result(self.__dfdu, tensor)
        return res

    def b_gu(self, u):
        tensor = type(u) is torch.Tensor
        res = self.__gu_t if tensor else self.__gu_np
        self.__u_t = torch.tensor(u)
        self.__u_t.reshape((self.params.n_batch, self.params.n_ctrl, 1))
        with torch.no_grad:
            return torch.bmm(res, self.__u_t)




