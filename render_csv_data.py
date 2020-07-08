from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import pandas as pd
from collections import namedtuple
from time import sleep

State = namedtuple('State', 'time qpos qvel act udd_state')

# finger_sample_1
# new_state = State(time=0, qpos=np.array([-np.pi/2 + 0.2, 0, 0+0.1]), qvel=np.array([0.0, 0, 0]), act=0, udd_state={})

if __name__ == "__main__":

    model = load_model_from_path("/home/daniel/Repos/OptimisationBasedControl/models/Acrobot.xml")
    sim = MjSim(model)

    df = pd.read_csv('/home/daniel/Repos/OptimisationBasedControl/Acrobot_offline.csv')
    ctrl_arr = df.to_numpy()
    ctrl_arr = np.delete(ctrl_arr, sim.data.ctrl.shape[0], 1)

    # finger_init = State(time=0, qpos=np.array([-np.pi/2 + 0.2, 0, 0+0.1]), qvel=np.array([0.0, 0, 0]), act=0, udd_state={})
    # cartpole_init = State(time=0, qpos=np.array([0, np.pi]), qvel=np.array([0, 0]), act=0, udd_state={})
    acrobot_init = State(time=0, qpos=np.array([0, 0]), qvel=np.array([0.0, 0]), act=0, udd_state={})

    sim.set_state(acrobot_init)
    viewer = MjViewer(sim)

    for time in range(len(ctrl_arr)):
        for control in range(sim.data.ctrl.shape[0]):
            sim.data.ctrl[control] = ctrl_arr[time][control]
        sim.step()
        viewer.render()
        sleep(0.01)
