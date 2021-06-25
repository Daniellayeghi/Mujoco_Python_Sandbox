from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
from tests.jacobian_compute import State
import time
import random 


if __name__ == "__main__":
    model = load_model_from_path("/home/daniel/Repos/OptimisationBasedControl/models/finger.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)

    state = State(time=0, qpos=np.array([0]), qvel=np.array([0]), act=0, udd_state={})
    sim.set_state(state)

    while True:
        viewer.render()
        sim.step()
        time.sleep(0.01)

        print(f"body {sim.data.contact[0].geom1} is in contact with {sim.data.contact[0].geom2}")
        sim.data.ctrl[1] = 1
