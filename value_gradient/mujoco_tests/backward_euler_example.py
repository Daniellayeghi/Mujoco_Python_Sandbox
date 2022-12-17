import mujoco
from mujoco.derivative import *
from dm_control.mujoco import Physics
import matplotlib.pyplot as plt
import torch
import numpy as np
import glfw
import time
import math
'''
This file tests the derivative bindings.
'''
m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator_sparse.xml")
# [dm, d2] = Physics.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml")
d = mujoco.MjData(m)
d.qpos = np.random.random(m.nq) * 5
print(m.nbody)

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


glfw.set_scroll_callback(glfw.get_current_context(), scroll)


if __name__ == "__main__":
    dfdu = MjDerivative(m, d, MjDerivativeParams(1e-6, Wrt.Ctrl, Mode.Fwd))
    df_du = list()

    while not glfw.window_should_close(glfw.get_current_context()):
        sim_start = d.time
        while(d.time - sim_start) < 1.0/60.0:
            step(m, d)
            d.ctrl[0] = - np.random.random(1) * .5
            res = dfdu.func()
            print(res)
            df_du.append(res.flatten())

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
