from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import pandas as pd
from collections import namedtuple

State = namedtuple('State', 'time qpos qvel act udd_state')

# finger_sample_1
# new_state = State(time=0, qpos=np.array([-np.pi/2 + 0.2, 0, 0+0.1]), qvel=np.array([0.0, 0, 0]), act=0, udd_state={})


if __name__ == "__main__":

    df = pd.read_csv('/home/daniel/Repos/OptimisationBasedControl/ctrl_finger.csv')
    ctrl_arr = df.to_numpy()
    ctrl_arr = np.delete(ctrl_arr, 2, 1)

    model = load_model_from_path("xmls/finger.xml")
    sim = MjSim(model)

    finger_init = State(time=0, qpos=np.array([-np.pi/2 + 0.2, 0, 0+0.1]), qvel=np.array([0.0, 0, 0]), act=0, udd_state={})
    cartpole_init = State(time=0, qpos=np.array([0, np.pi]), qvel=np.array([0, 0]), act=0, udd_state={})

    sim.set_state(finger_init)
    viewer = MjViewer(sim)
    for time in range(len(ctrl_arr)):
        for control in range(sim.data.ctrl.shape[0]):
            sim.data.ctrl[control] = ctrl_arr[time][control]
        sim.step()
        viewer.render()
