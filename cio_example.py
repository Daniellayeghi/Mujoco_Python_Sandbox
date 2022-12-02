import mujoco
import matplotlib.pyplot as plt
import numpy as np
import math

m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator_sparse.xml")
d = mujoco.MjData(m)

dist = []
ci_cst = []
pos_a = np.linspace(-5, -1, 100)
pos_u = np.linspace(5, 1, 100)
for i in range(pos_u.size):
    dist.append(math.fabs(pos_a[i] - pos_u[i]))
    d.qpos = np.array([pos_a[i], pos_u[i]])
    d.qvel = np.array([0, 0])
    d.qacc = np.array([0, 1])
    mujoco.mj_inverse(m, d)
    ci_cst.append(d.qfrc_inverse[1] * dist[i])

plt.plot(dist, ci_cst)
plt.show()
