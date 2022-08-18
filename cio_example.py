import random
import mujoco
from mujoco import derivative
import matplotlib.pyplot as plt
import numpy as np
import glfw
import time
import math

m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator_sparse.xml")
d = mujoco.MjData(m)
d.qpos = np.array([2, .5])
d.qvel = np.array([0, 0])
d.qacc = np.array([0, 0])
des_pos = 0
dt = 0.01

# Computing derivatives
deriv = lambda x_1, x_2, delta: (x_2 - x_1) / dt
xd_2 = deriv(d.qpos[1], des_pos, dt)
xdd_2 = deriv(d.qvel[1], xd_2, dt)

d.qpos = np.array([2, des_pos])
d.qvel = np.array([0, xd_2])
d.qacc = np.array([0, xdd_2])

mujoco.mj_inverse(m, d)
print(d.qfrc_inverse)

