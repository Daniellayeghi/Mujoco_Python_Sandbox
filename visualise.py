from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
from tests.jacobian_compute import State
import time
import random


if __name__ == "__main__":
    model = load_model_from_path("/home/daniel/Repos/OptimisationBasedControl/models/contact_comp/planar_push.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)

    state = State(time=0, qpos=np.array([0]), qvel=np.array([0]), act=0, udd_state={})
    sim.set_state(state)

    while True:
        viewer.render()
        sim.step()
        time.sleep(0.01)
        
        for i in range(sim.data.ncon):
            print(f"contact {i} body {model.geom_bodyid[sim.data.contact[i].geom1]} is in contact with {model.geom_bodyid[sim.data.contact[i].geom2]}")
        sim.data.ctrl[0] = -0.2
