from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
from collections import namedtuple

State = namedtuple('State', 'time qpos qvel act udd_state')

# finger_sample_1
# new_state = State(time=0, qpos=np.array([-np.pi/2 + 0.2, 0, 0+0.1]), qvel=np.array([0.0, 0, 0]), act=0, udd_state={})


if __name__ == "__main__":
    model = load_model_from_path("xmls/finger.xml")
    sim = MjSim(model)
    finger_init = State(time=0, qpos=np.array([-np.pi/2 + 0.2, 0, 0+0.1]), qvel=np.array([0.0, 0, 0]), act=0, udd_state={})
    cartpole_init = State(time=0, qpos=np.array([0, np.pi]), qvel=np.array([0, 0]), act=0, udd_state={})
    sim.set_state(finger_init)
    viewer = MjViewer(sim)

    cartpole_controls = np.load("working_controls_finger_1000.0_25000_200_500_500_0.9.npy")
    for time in range(len(cartpole_controls)):
        for control in range(sim.data.ctrl.shape[0]):
            sim.data.ctrl[control] = cartpole_controls[time][control]
        sim.step()
        viewer.render()
