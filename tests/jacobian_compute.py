import mujoco
import numpy as np
import numdifftools as nd
from scipy.optimize import approx_fprime
from mujoco.derivative import *


m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/2link.xml")
d = mujoco.MjData(m)
d_cp = mujoco.MjData(m)
dfdu = MjDerivative(m, d, MjDerivativeParams(1e-6, Wrt.Ctrl, Mode.Fwd))


def copy_data(d_src, d_target):
    d_target.qpos = d_src.qpos
    d_target.qvel = d_target.qvel
    d_target.qacc = d_target.qacc
    d_target.qfrc_applied = d_target.qfrc_applied
    d_target.xfrc_applied = d_target.xfrc_applied
    d_target.ctrl = d_target.ctrl
    mujoco.mj_forward(m, d_target)


def fwd_u(u):
    d_cp.ctrl[1] = u
    mujoco.mj_step(m, d_cp)
    res = np.copy(d_cp.qpos)
    copy_data(d, d_cp)
    return res[0]


def fwd_x(x):
    d.qpos = x[:m.nq]
    d.qvel = x[m.nq:]
    mujoco.mj_step(m, d)
    return np.vstack((d.qpos, d.qvel, d.qacc)).flatten()


if __name__ == "__main__":
    d.qpos, d.qvel, n_full, epsilon = 0, 0, m.nq * 3, 1e-6
    d_cp.qpos, d_cp.qvel, = 0, 0

    u = np.zeros_like(d.ctrl)
    dxdu_mine = dfdu.func()

    d = mujoco.MjData(m)
    d.qpos, d.qvel, n_full, epsilon = 0, 0, m.nq * 3, 1e-6
    d_cp.qpos, d_cp.qvel, = 0, 0

    jac_op = nd.Jacobian(lambda x: fwd_u(x), method='forward', step=epsilon, order=1, step_ratio=1)
    dxdu_nd = jac_op(0)

    print(f"[dxdu Error]: {np.square(np.sum(dxdu_nd[:-1] - dxdu_mine))}")

    hess_op = [nd.Hessian(lambda x: fwd_u(x)[m], method='forward') for m in range(n_full)]
    dx2_d2u = np.vstack([hess_op[i](u) for i in range(len(hess_op))])

    print(f"dx2_d2u: {dx2_d2u} and gauss dx2_d2u: {np.outer(2 * dxdu_mine.T, dxdu_mine)}")
