import mujoco
from mujoco.derivative import *
from dm_control.mujoco import Physics
import matplotlib.pyplot as plt
from utilities.torch_utils import np_to_tensor
import torch
import torch.nn.functional as Func
from torch.autograd.functional import jacobian
import numpy as np
import glfw
import time
import math
'''
This file tests the derivative bindings.
'''
m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml")
d = mujoco.MjData(m)
d_cp = mujoco.MjData(m)

ctx = mujoco.GLContext(1200, 1200)
ctx.make_current()
glfw.swap_interval(1)
glfw.show_window(ctx._context)
cam = mujoco.MjvCamera()
pert = mujoco.MjvPerturb()
con = mujoco.MjrContext(m, 100)
scn = mujoco.MjvScene(m, 1000)
opt = mujoco.MjvOption()
mujoco.mjv_defaultCamera(cam)
mujoco.mjv_defaultOption(opt)
mujoco.mjv_updateScene(m, d, opt, pert, cam, 1, scn)


def copy_data(d_src, d_target):
    d_target.qpos = d_src.qpos
    d_target.qvel = d_src.qvel
    d_target.qacc = d_src.qacc
    d_target.qfrc_applied = d_src.qfrc_applied
    d_target.xfrc_applied = d_src.xfrc_applied
    d_target.ctrl = d_src.ctrl
    mujoco.mj_forward(m, d_target)


def scroll(window, x_off, y_off):
    mujoco.mjv_moveCamera(m, 5, 0, -0.05*y_off, scn=scn, cam=cam)


def control_cb(model: mujoco.MjModel, data: mujoco.MjData):
    # Implements a simple double integrator policy centered at 0
    # d.ctrl[0] = -d.qpos[0] - math.sqrt(3) * d.qvel[0]
    pass


def dummy_control_cb(model: mujoco.MjModel, data: mujoco.MjData):
    pass


def step(m, d):
    mujoco.set_mjcb_control(control_cb)
    mujoco.mj_step(m, d)
    mujoco.set_mjcb_control(dummy_control_cb)


def value_func(x):
    q, qd = x[0], x[1]
    value = 1.732 * q**2 + 2*q*qd + 1.732 * qd**2
    return value


def copy_state_to_tensor(x_np):
    x = np_to_tensor(x_np, False, True)
    return x


def project(x_xd, dvdx, loss=lambda x_xd: (x_xd @ x_xd)):
    x, xd = x_xd[0:2], x_xd[1:3]
    norm = torch.sqrt(dvdx @ dvdx + 1e-6)
    proj = Func.relu((dvdx@xd) + (50 * loss(x_xd)))
    xd_next = xd - (dvdx/norm * proj)
    return xd_next


glfw.set_scroll_callback(glfw.get_current_context(), scroll)

done = False
if __name__ == "__main__":
    dfdx = MjDerivative(m, d, MjDerivativeParams(1e-6, Wrt.State, Mode.Fwd))
    d.qpos = -10
    df_du = list()
    while not glfw.window_should_close(glfw.get_current_context()) and not done:
        sim_start = d.time
        while(d.time - sim_start) < 1.0/60.0:
            copy_data(d, d_cp)
            x_xd = copy_state_to_tensor(np.hstack((d.qpos, d.qvel, d.qacc)))
            x = copy_state_to_tensor(np.hstack((d.qpos, d.qvel)))
            dvdx = jacobian(value_func, x)
            xd_next = project(x_xd, dvdx)

            # in the case of the point mass the step in the cartesian
            # step is 1 to 1 with the "external point" mass (NO IK)
            d_cp.qpos, d_cp.qvel, d_cp.qacc = d.qpos + d_cp.qvel * 0.01, d.qvel + xd_next[1].detach().numpy() * 0.01, xd_next[1].detach().numpy()

            mujoco.mj_inverse(m, d_cp)
            d.ctrl = d_cp.qfrc_inverse
            mujoco.mj_step(m, d)
            df_du.append(d.qpos[0])
            if(d.qpos**2) < 1e-4:
                done = True

        time.sleep(0.01)
        view = mujoco.MjrRect(0, 0, 0, 0)
        view.width, view.height = glfw.get_framebuffer_size(glfw.get_current_context())
        mujoco.mjv_updateScene(m, d, opt, pert, cam, 7, scn)
        mujoco.mjr_render(viewport=view, scn=scn, con=con)
        glfw.swap_buffers(glfw.get_current_context())
        glfw.poll_events()

    plt.plot(df_du)
    plt.title("dpos/du")
    plt.show()
    ctx.free()
    con.free()
