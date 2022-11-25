import mujoco
from mujoco.derivative import *
from .torch_utils import *
from torch.autograd import Function
from .torch_device import device
from collections import namedtuple


SimulationParams = namedtuple(
    'SimulationParams', 'nqva, nqv, nq, nv, nu, nee, nsim, ntime'
)


_mj_attrs = None


class __MJAtrributes():
    def __init__(self, m: mujoco.MjModel, sim_params: SimulationParams):
        self._m = m
        self._d = mujoco.MjData(m)
        self._dfdx = MjDerivative(m, self._d, MjDerivativeParams(1e-6, Wrt.State, Mode.Fwd))
        self._dfdu = MjDerivative(m, self._d, MjDerivativeParams(1e-6, Wrt.Ctrl, Mode.Fwd))
        self._dfinvdx_xd = MjDerivative(m, self._d, MjDerivativeParams(1e-6, Wrt.State, Mode.Inv))
        self._nstate = self._m.nv + self._m.nq
        self._nfull_state = self._nstate + self._m.nq

        self._sim_params = sim_params
        self.n_sim = sim_params.nsim
        self.n_time = sim_params.ntime
        self.n_ee = sim_params.nee
        self.nqv = sim_params.nqv

        # Buffers
        self._qfrc_t = torch.zeros(
            (self.n_time, self.n_sim, 1, self._m.nv), dtype=torch.float
        ).requires_grad_().to(device)

        self._dfinvdx_xd_t = torch.zeros(
            (self.n_time, self.n_sim, self._m.nv, self._nfull_state), dtype=torch.float
        ).requires_grad_().to(device)

    # Setters
    def _set_q_qd_qdd(self, pos, vel, acc):
        self._d.qpos, self._d.qvel, self._d.qacc = pos, vel, acc

    def _set_q_qd(self, pos, vel):
        self._d.qpos, self._d.qvel = pos, vel

    def set_q_qd_qdd_array(self, x_xd_tensor):
        tensor = x_xd_tensor.flatten()
        self._set_q_qd_qdd(
            tensor[:self._m.nq], tensor[self._m.nq:self._nstate], tensor[self._nstate:]
        )

    def set_q_qd_array(self, x_tensor):
        tensor = x_tensor.flatten()
        self._set_q_qd(
            tensor[:self._m.nq], tensor[self._m.nq:self._nstate]
        )

    def detach(self):
        self._qfrc_t.detach()
        self._dfinvdx_xd_t.detach()

    def attach(self):
        self._qfrc_t.requires_grad_()
        self._dfinvdx_xd_t.requires_grad_()


def torch_mj_set_attributes(m: mujoco.MjModel, sim_params: SimulationParams):
    global _mj_attrs
    _mj_attrs = __MJAtrributes(m, sim_params)


def torch_mj_detach():
    global _mj_attrs
    _mj_attrs.detach()


def torch_mj_attach():
    global _mj_attrs
    _mj_attrs.attach()


class torch_mj_inv(Function):
    @staticmethod
    def forward(ctx, x_xd):
        x_xd_np = tensor_to_np(x_xd)
        ctx.save_for_backward(x_xd)
        res = _mj_attrs._qfrc_t

        time, nsim, r, c = x_xd.shape
        for t in range(time):
            for s in range(nsim):
                _mj_attrs.set_q_qd_qdd_array(x_xd_np[t, s, :, :])
                mujoco.mj_inverse(_mj_attrs._m, _mj_attrs._d)
                res[t, s, :, :] = np_to_tensor(_mj_attrs._d.qfrc_inverse, x_xd.device).view(res[t, s, :, :].shape)

        return res

    @staticmethod
    def backward(ctx, grad_output):
        x_xd, = ctx.saved_tensors
        x_xd_np = tensor_to_np(x_xd)
        time, nsim, r, c = x_xd.shape
        for t in range(time):
            for s in range(nsim):
                _mj_attrs.set_q_qd_qdd_array(x_xd_np[t, s, :, :])
                _mj_attrs._dfinvdx_xd_t[t, s, :, :] = np_to_tensor(
                _mj_attrs._dfinvdx_xd.func(), _mj_attrs._dfinvdx_xd_t.device
            )

        return grad_output * _mj_attrs._dfinvdx_xd_t
