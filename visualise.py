from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
from tests.jacobian_compute import State
import time
import random 

class QRCost(object):
    def __init__(self, q: np.array, r: np.array, ref: np.array):
        self._Q = q
        self._R = r
        self._ref = ref

    def cost_function(self, state, ctrl: np.array):
        err = self._ref - state
        state_cost = err.dot(self._Q).dot(err)
        ctrl_reg   = ctrl.dot(self._R).dot(ctrl)
        return state_cost + ctrl_reg


if __name__ == "__main__":
    model = load_model_from_path("/home/daniel/Repos/OptimisationBasedControl/models/point_mass.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)

    state = State(time=0, qpos=np.array([0]), qvel=np.array([0]), act=0, udd_state={})
    sim.set_state(state)

    cost = QRCost(
        np.diagflat(np.array([5000, 500 * 0.01])),
        np.diagflat(np.array([1])),
        np.array([1.46152155e-17, 0.00000000e+00, 1.19342291e-01, 0])
    )
    pos = 0
    while True:
        viewer.render()
        sim.step()
        print(cost.cost_function(
            np.append(sim.data.xipos[1], sim.data.qvel[0]), sim.data.ctrl
        ))
        state = State(time=0, qpos=np.array([pos]), qvel=np.array([0]), act=0, udd_state={})
        sim.set_state(state)
        time.sleep(0.01)
        pos = pos-0.01
