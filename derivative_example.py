import mujoco
from mujoco import derivative
import matplotlib.pyplot as plt
import numpy as np
import glfw
import time
import math
'''
This file tests the derivative bindings.
'''
m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml")
d = mujoco.MjData(m)
d.qpos = np.random.random(m.nq) * 5

ctx = mujoco.GLContext(1200, 1200)
ctx.make_current()
glfw.swap_interval(1)
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


glfw.set_scroll_callback(glfw.get_current_context(), scroll)


if __name__ == "__main__":
    print(m.nD)
    d_vec = derivative.MjDataVecView(m, d)
    params = derivative.MjDerivativeParams(1e-6, derivative.WRT.CTRL)
    mj_deriv = derivative.MjDerivative(m, params)
    print(mj_deriv.dyn_derivative(d_vec))
    mujoco.mj_step(m, d)
    print(mj_deriv.dyn_derivative(d_vec))
    mujoco.mj_step(m, d)
    print(mj_deriv.dyn_derivative(d_vec))
    mujoco.mj_step(m, d)
    print(mj_deriv.dyn_derivative(d_vec))

    u_d = list()

    # Control callback
    def control_cb(model: mujoco.MjModel, data: mujoco.MjData):
        # Implements a simple double integrator policy centered at 0
        d.ctrl[0] = -d.qpos - math.sqrt(3) * d.qvel

    def dummy_control_cb(model: mujoco.MjModel, data: mujoco.MjData):
        pass

    def step(m, d):
        mujoco.set_mjcb_control(control_cb)
        mujoco.mj_step(m, d)
        mujoco.set_mjcb_control(dummy_control_cb)


    while not glfw.window_should_close(glfw.get_current_context()):
        sim_start = d.time
        while(d.time - sim_start) < 1.0/60.0:
            step(m, d)
            print(mj_deriv.dyn_derivative(d_vec))

        time.sleep(0.001)
        view = mujoco.MjrRect(0, 0, 0, 0)
        view.width, view.height = glfw.get_framebuffer_size(glfw.get_current_context())
        mujoco.mjv_updateScene(m, d, opt, pert, cam, 7, scn)
        mujoco.mjr_render(viewport=view, scn=scn, con=con)
        glfw.swap_buffers(glfw.get_current_context())
        glfw.poll_events()

    plt.show()
    ctx.free()
    con.free()