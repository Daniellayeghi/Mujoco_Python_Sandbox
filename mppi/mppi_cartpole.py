from utils.utilities import State
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mppi.mppi_control import MPPI


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

    def running_cost(self, state, variance, control, control_dist):
        QR_cost = (1 - 1/variance)/2 + control_dist.T * self._R + control_dist + \
                  control.T * self._R * control + 0.5 * control.T * self._R * control

        return (500 * (1 - np.cos(state.qpos[1])) ** 2 + 800 * state.qvel[1] ** 2 * 0.01) + \
               (500 * (state.qpos[0]) ** 2 + 400 * state.qvel[0] ** 2 * 0.01) + QR_cost


if __name__ == "__main__":
    model = load_model_from_path("xmls/cartpole.xml")
    sim = MjSim(model)
    plant = MjSim(model)

    new_state = State(time=0, qpos=np.array([0, np.pi]), qvel=np.array([0, 0]), act=0, udd_state={})
    plant.set_state(new_state)

    Params = {"R": 500.0, "Lambda": 2000, "Samples": 50, "Horizon": 100, "Time": 250, "Variance": 0.999}

    cost = Cost(None, np.eye(sim.data.ctrl.shape[0]) * Params["R"], None, Params["Lambda"])

    pi = MPPI(sim, Params["Samples"], Params["Horizon"], cost, Params["Variance"], plant, Params["Time"])
    pi.simulate(viewer=None)

    np.save(f'mppi/results/working_controls_cartpole_{Params["R"]}_'
            f'                                       {Params["Lamda"]}_'
            f'                                       {Params["Samples"]}_'
            f'                                       {Params["Horizon"]}_'
            f'                                       {Params["Time"]}_'
            f'                                       {Params["variance"]}.npy',
                                                     pi.plant_control[:])

    viewer = MjViewer(plant)
    rec = input("Visualise ?")
    print("Visualising")
    plant.set_state(new_state)

    for control in range(len(pi.plant_control)):
        plant.data.ctrl[0] = pi.plant_control[control][0]
        plant.step()
        viewer.render()

