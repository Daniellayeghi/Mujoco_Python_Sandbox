from mppi_control import MPPI
from jacobian_compute import State
import numpy as np
import matplotlib.pyplot as plt
from mujoco_py import load_model_from_path, MjSim, MjViewer
from tempfile import TemporaryFile
from collections import namedtuple


class Cost(object):
    def __init__(self,
                 Q,
                 R,
                 goal,
                 value_scale):

        self._Q = Q
        self._R = R
        self._goal = goal
        self.value_scale = value_scale

    def running_cost(self, state):
        return (100 * (1 - np.cos(state.qpos[1])) ** 2 + 0.2 * state.qvel[1] ** 2) * 0.01 + \
               (50 * (state.qpos[0]) ** 2 + 0.2 * state.qvel[0] ** 2) * 0.01


if __name__ == "__main__":
    model = load_model_from_path("xmls/cartpole.xml")
    sim = MjSim(model)
    plant = MjSim(model)

    new_state = State(time=0, qpos=np.array([0, np.pi]), qvel=np.array([0, 0]), act=0, udd_state={})
    plant.set_state(new_state)
    cost = Cost(None, None, None, 10)
    pi = MPPI(sim, 10, 200, cost, 0.25, plant, 1250)
    pi.simulate()
    np.save("control.npy", pi.plant_control[:])

    rec = input("Visualise ?")
    print("Visualising")
    plant.set_state(new_state)
    viewer = MjViewer(plant)

    for control in range(len(pi.plant_control)):
        plant.data.ctrl[0] = pi.plant_control[control][0]
        plant.step()
        viewer.render()

