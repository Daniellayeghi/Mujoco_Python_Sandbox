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
        self._data = mujoco.MjData(m)
        self._dfdx = MjDerivative(m, self._data, MjDerivativeParams(1e-6, Wrt.State, Mode.Fwd))
        self._dfdu = MjDerivative(m, self._data, MjDerivativeParams(1e-6, Wrt.Ctrl, Mode.Fwd))
        self._dfinvdx = MjDerivative(m, self._data, MjDerivativeParams(1e-6, Wrt.State, Mode.Inv))

        self._qfrc_t = torch.zeros(
            (self.params.n_batch, self.params.n_vel), dtype=torch.double
        ).requires_grad_()

        self._dxdt_t = torch.zeros(
            (self.params.n_batch, self.params.n_vel * 2), dtype=torch.double
        ).requires_grad_()

        self._dfinvdx_t = torch.zeros(
            (self.params.n_batch, self.params.n_full_state * self.params.n_vel), dtype=torch.double
        ).requires_grad_()

        self._dfdx_t = torch.zeros(
            (self.params.n_batch, self.params.n_state**2), dtype=torch.double
        ).requires_grad_()

        self._dfdu_t = torch.zeros(
            (self.params.n_batch, self.params.n_ctrl*self.params.n_state), dtype=torch.double
        ).requires_grad_()

        self._gu_t = torch.zeros(
            (self.params.n_batch, self.params.n_state, self.params.n_ctrl), dtype=torch.double
        )

        self._u_t = torch.zeros(
            (self.params.n_batch, self.params.n_ctrl), dtype=torch.double
        ).requires_grad_()

        self._gu_t[:, self.params.idx_g_act[0]:self.params.idx_g_act[1], :] = 1
        self._gu_t.requires_grad_()

        self._qfrc_np = self._qfrc_t.detach().numpy().astype('float64')
        self._dfinvdx_np = self._dfinvdx_t.detach().numpy().astype('float64')
        self._dxdt_np = self._dxdt_t.detach().numpy().astype('float64')
        self._dfdx_np = self._dfdx_t.detach().numpy().astype('float64')
        self._dfdu_np = self._dfdu_t.detach().numpy().astype('float64')
        self._gu_np = self._gu_t.detach().numpy().astype('float64')

    def _reset_inverse(self):
        self._data.qfrc_inverse[:] = 0

    @staticmethod
    def _get_result(operation, tensor_flag):
        if tensor_flag:
            return torch.Tensor(operation.func().flatten())
        else:
            return operation.func().flatten()

    def _mj_set_ctrl(self, u):
        self._data.ctrl = u

    def _mj_set_x_full_decomp(self, pos, vel, acc):
        self._data.qpos, self._data.qvel, self._data.qacc = pos, vel, acc

    def _mj_set_state(self, pos, vel):
        self._data.qpos, self._data.qvel = pos, vel

    def _mj_set_x_full(self, x):
        x = x.flatten()
        self._mj_set_x_full_decomp(
            x[:self.params.n_pos], x[self.params.n_pos:self.params.n_state], x[self.params.n_state:]
        )

    def _mj_set_x(self, x):
        x = x.flatten()
        self._mj_set_state(x[:self.params.n_pos], x[self.params.n_pos:self.params.n_state])

    def _mj_set_x_decomp_ctrl(self, pos, vel, u):
        self._mj_set_ctrl(u)
        self._mj_set_state(pos, vel)

    def _mj_set_x_ctrl(self, x, u):
        self._mj_set_ctrl(u)
        self._mj_set_state(x[:self.params.n_pos], x[self.params.n_pos:self.params.n_state])

    def f(self, tensor=True):
        mujoco.mj_step(self.__model, self._data)
        res = np.hstack((self._data.qvel, self._data.qacc)).flatten()
        if tensor:
            return torch.Tensor(res)
        else:
            return res

    def finv(self, tensor=True):
        mujoco.mj_inverse(self.__model, self._data)
        res = self._data.qfrc_inverse.flatten()
        if tensor:
            return torch.Tensor(res)
        else:
            return res

    def b_qfrc_decomp(self, pos, vel, acc):
        tensor = type(pos) is torch.Tensor
        res = self._qfrc_t if tensor else self._qfrc_np
        with torch.no_grad():
            for i in range(pos.shape[0]):
                self._mj_set_x_full_decomp(pos[i, :], vel[i, :], acc[i, :])
                res[i, :] = self.finv(tensor)
        return res

    def b_qfrcs(self, x_full):
        tensor = type(x_full) is torch.Tensor
        res = self._qfrc_t if tensor else self._qfrc_np
        with torch.no_grad():
            for i in range(x_full.shape[0]):
                self._mj_set_x_full(x_full[i, :])
                res[i, :] = self.finv(tensor)

        self._reset_inverse()
        return res

    def b_dxdt_decomp(self, pos, vel, u):
        tensor = type(pos) is torch.Tensor
        res = self._dfdx_t if tensor else self._dfdx_np
        with torch.no_grad():
            for i in range(pos.shape[0]):
                self._mj_set_x_decomp_ctrl(pos[i, :], vel[i, :], u[i, :])
                res[i, :] = self.f(tensor)
        return res

    def b_dxdt(self, x, u):
        tensor = type(x) is torch.Tensor
        res = self._dxdt_t if tensor else self._dxdt_np
        with torch.no_grad():
            for i in range(x.shape[0]):
                self._mj_set_x_ctrl(x[i, :], u[i, :])
                res[i, :] = self.f(tensor)
        return res

    def b_dfinvdx_full(self, x_full):
        tensor = type(x_full) is torch.Tensor
        res = self._dfinvdx_t if tensor else self._dfinvdx_np
        with torch.no_grad():
            for i in range(x_full.shape[0]):
                self._mj_set_x_full(x_full[i, :])
                res[i, :] = self._get_result(self._dfinvdx, tensor)
        return res

    def b_dfinvdx_full_decomp(self, pos, vel, acc):
        tensor = type(pos) is torch.Tensor
        res = self._dfinvdx_t if tensor else self._dfinvdx_np
        with torch.no_grad():
            for i in range(pos.shape[0]):
                self._mj_set_x_full_decomp(pos[i, :], vel[i, :], acc[i, :])
                res[i, :] = self._get_result(self._dfinvdx, tensor)
        return  res

    def b_dfdx(self, x, u):
        tensor = type(x) is torch.Tensor
        with torch.no_grad():
            res = self._dfdx_t if tensor else self._dfdx_np
            for i in range(x.shape[0]):
                self._mj_set_x_ctrl(x[i, :], u[i, :])
                res[i, :] = self._get_result(self._dfdx, tensor)
        return res

    def b_dfdx_decomp(self, pos, vel, u):
        tensor = type(pos) is torch.Tensor
        res = self._dfdx_t if tensor else self._dfdx_np
        with torch.no_grad():
            for i in range(pos.shape[0]):
                self._mj_set_x_decomp_ctrl(pos[i, :], vel[i, :], u[i, :])
                res[i, :] = self._get_result(self._dfdx, tensor)
        return res

    def b_dfdu(self, x, u):
        tensor = type(x) is torch.Tensor
        res = self._dfdu_t if tensor else self._dfdu_np
        with torch.no_grad():
            for i in range(x.shape[0]):
                self._mj_set_x_ctrl(x[i, :], u[i, :])
                res[i, :] = self._get_result(self._dfdu, tensor)
        return res

    def b_gu(self, u):
        tensor = type(u) is torch.Tensor
        res = self._gu_t
        self._u_t = torch.tensor(u)
        self._u_t.reshape((self.params.n_batch, self.params.n_ctrl, 1))
        with torch.no_grad:
            return torch.bmm(res, self._u_t) if tensor else torch.bmm(res, self._u_t).numpy()




