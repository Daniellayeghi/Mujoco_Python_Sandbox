import mujoco
from mujoco.derivative import *
from .torch_utils import *
from torch.autograd import Function


_mj_attrs = None


class __MJAtrributes():
    def __init__(self, m: mujoco.MjModel, batch_size):
        self.m = m
        self.d = mujoco.MjData(m)
        self._dfdx = MjDerivative(m, self.d, MjDerivativeParams(1e-6, Wrt.State, Mode.Fwd))
        self._dfdu = MjDerivative(m, self.d, MjDerivativeParams(1e-6, Wrt.Ctrl, Mode.Fwd))
        self._dfinvdx_xd = MjDerivative(m, self.d, MjDerivativeParams(1e-6, Wrt.State, Mode.Inv))
        self._n_state = self.m.nq + self.m.nv

        # Buffers
        self._qfrc_t = torch.zeros(
            (batch_size, self.m.nv), dtype=torch.float
        ).requires_grad_()

        self._dfinvdx_xd_t = torch.zeros(
            (batch_size, (self._n_state + self.m.nv) * self.m.nv), dtype=torch.float
        ).requires_grad_()

    # Setters
    def _set_q_qd_qdd(self, pos, vel, acc):
        self.d.qpos, self.d.qvel, self.d.qacc = pos, vel, acc

    def _set_q_qd(self, pos, vel):
        self.d.qpos, self.d.qvel = pos, vel

    def set_q_qd_qdd_array(self, x_xd_tensor):
        tensor = x_xd_tensor.flatten()
        self._set_q_qd_qdd(
            tensor[:self.m.nq], tensor[self.m.nq:self._n_state], tensor[self._n_state:]
        )

    def set_q_qd_array(self, x_tensor):
        tensor = x_tensor.flatten()
        self._set_q_qd(
            tensor[:self.m.nq], tensor[self.m.nq:self._n_state]
        )


def torch_mj_set_attributes(m: mujoco.MjModel, bs):
    global _mj_attrs
    _mj_attrs = __MJAtrributes(m, bs)


class torch_mj_inv(Function):
    @staticmethod
    def forward(ctx, x_xd):
        x_xd_np = tensor_to_np(x_xd)
        ctx.save_for_backward(x_xd)

        # Rollout batch
        for i in range(len(x_xd_np)):
            _mj_attrs.set_q_qd_qdd_array(x_xd_np[i, :])
            mujoco.mj_inverse(_mj_attrs.m, _mj_attrs.d)
            _mj_attrs._qfrc_t[i, :] = np_to_tensor(_mj_attrs.d.qfrc_inverse, _mj_attrs._qfrc_t.device)

        return _mj_attrs._qfrc_t

    @staticmethod
    def backward(ctx, grad_output):
        x_xd_np, = ctx.saved_tensors
        for i in range(len(x_xd_np)):
            _mj_attrs.set_q_qd_qdd_array(x_xd_np[i, :])
            _mj_attrs._dfinvdx_xd_t[i, :] = np_to_tensor(_mj_attrs._dfinvdx_xd.func(), _mj_attrs._dfinvdx_xd_t.device)
        return grad_output * _mj_attrs._dfinvdx_xd_t


