from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
from tests.jacobian_compute import State
import time
import random 


if __name__ == "__main__":
    model = load_model_from_path("/home/daniel/Repos/OptimisationBasedControl/models/point_mass.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)
    print(sim.data.qvel)
    #state = State(time=0, qpos=sim.data.qpos, qvel=np.random.rand(1,8), act=0, udd_state={})
    #sim.set_state(state)
    while True:
        viewer.render()
        sim.step()
        #state = State(time=0, qpos=sim.data.qpos, qvel=np.random.rand(1,8)*0.025, act=0, udd_state={})
        #sim.set_state(state)       
        ctrl = random.random()
        print(ctrl)
        sim.data.ctrl[0] = ctrl
        time.sleep(0.01)
