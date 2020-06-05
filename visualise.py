from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
from tests.jacobian_compute import State

if __name__ == "__main__":
    model = load_model_from_path("xmls/finger.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)

    state = State(time=0, qpos=np.array([-np.pi/2 + 0.5, -0.8, 0+0.1]), qvel=np.array([0.0, 0, 0]), act=0, udd_state={})
    sim.set_state(state)

    while True:
        viewer.render()
        sim.step()
