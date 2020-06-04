from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
from collections import namedtuple

State = namedtuple('State', 'time qpos qvel act udd_state')

if __name__ == "__main__":
    model = load_model_from_path("xmls/cartpole.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)
    new_state = State(time=0, qpos=np.array([0, 1]), qvel=np.array([0, 0]), act=0, udd_state={})
    sim.set_state(new_state)
    while True:
        viewer.render()
        sim.step()
        print(sim.data.xipos)