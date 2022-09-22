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
    d_target.qvel = d_src.qvel
    d_target.qacc = d_src.qacc
    d_target.qfrc_applied = d_src.qfrc_applied
    d_target.xfrc_applied = d_src.xfrc_applied
    d_target.ctrl = d_src.ctrl
    mujoco.mj_forward(m, d_target)


def fwd_u(u):
    d_cp.ctrl = u
    mujoco.mj_step(m, d_cp)
    res = np.vstack((d_cp.qpos, d_cp.qvel))
    copy_data(d, d_cp)
    return res.flatten()


def fwd_x(x):
    d.qpos = x[:m.nq]
    d.qvel = x[m.nq:]
    mujoco.mj_step(m, d)
    return np.vstack((d.qpos, d.qvel, d.qacc)).flatten()


if __name__ == "__main__":
    d.qpos, d.qvel, n_full, epsilon = [10, 10], [0, 0], m.nq * 2, 1e-6
    d_cp.qpos, d_cp.qvel = [10, 10], [0, 0]

    u = np.zeros_like(d.ctrl)
    dxdu_mine = dfdu.func()

    jac_op = nd.Jacobian(
        lambda x: fwd_u(x), method='forward', step=epsilon, order=0, step_ratio=1
    )

    dxdu_nd = jac_op(u.T)
    print(f"[dxdu Error]: {np.square(np.sum(dxdu_nd - dxdu_mine))} \n")

    u_rand = np.array([0, 0])

    hess_ops = []
    for i in range(n_full):
        op = nd.Hessian(
            lambda u: fwd_u(u)[i], method='forward', step=epsilon, order=0, step_ratio=0
        )
        hess_ops.append(op)

    dx2_d2u = []
    for i in range(len(hess_ops)):
        dx2_d2u.append(hess_ops[i](u_rand))

    dx2_d2u = np.vstack(dx2_d2u)
    print(f"dx2_d2u:\n {dx2_d2u} \n gauss:\n {dxdu_mine.T @ dxdu_mine}")
