from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
from utils.utilities import State


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
        QR_cost = (1 - 1/variance)/2 + control_dist.T.dot(self._R).dot(control_dist) + \
                  control.T.dot(self._R).dot(control) + 0.5 * control.T.dot(self._R).dot(control)

        return (800 * (1 - np.cos(state.qpos[2]))**2 + 50 * state.qvel[2]**2 * 0.01) + QR_cost


class MPPI(object):
    def __init__(self,
                 sim,
                 samples: int,
                 horizon: int,
                 cost_function: Cost,
                 variance,
                 plant,
                 time_horizon):

        self._sim = sim
        self._samples  = samples
        self._horizon  = horizon
        self._variance = variance
        self._cost_func  = cost_function
        self._num_actions = self._sim.data.ctrl.shape[0]
        self._delta_ctrl = [np.full((horizon, samples), 0.0) for _ in range(self._num_actions)]
        self._ctrl = [np.full((horizon, 1), 0.0) for _ in range(self._num_actions)]
        self._cost       = np.full((horizon, 1), 0.0)
        self._cost_avg   = np.full((time_horizon, 1), 0.0)
        self._sample_cost = np.full((samples, 1), 0.0)

        self._plant = plant
        self._time_horizon  = time_horizon
        self._plant_cost    = np.full((time_horizon, 1), 0.0)
        self._plant_state   = [State for _ in range(time_horizon)]
        self.plant_control  = [np.zeros((self._num_actions, 1)) for _ in range(time_horizon)]

    def total_entropy(self, delta_ctrl_samples):
        num = np.zeros((self._num_actions, 1))
        den = 0.0

        for sample in range(self._samples):
            ctrl_array = np.array([delta_ctrl_samples[i][sample] for i in range(self._num_actions)])
            num += np.exp(-(1/self._cost_func.value_scale * self._sample_cost[sample])) * ctrl_array.reshape(self._num_actions, 1)
            den += np.exp(-(1/self._cost_func.value_scale * self._sample_cost[sample]))

        return num/den

    def compute_controls(self):
        self._sample_cost.fill(0)
        for sample in range(self._samples):
            self._sim.set_state(self._plant.get_state())
            for time in range(self._horizon - 1):
                for controls in range(self._num_actions):
                    self._delta_ctrl[controls][time][sample] = np.random.uniform(-1, 1, 1)[0] * self._variance
                    self._sim.data.ctrl[controls] = self._ctrl[controls][time][0] + self._delta_ctrl[controls][time][sample]

                self._sim.step()

                self._sample_cost[sample] = self._sample_cost[sample] + \
                                            self._cost_func.running_cost(self._sim.get_state(),
                                                                         self._variance,
                                                                         np.array([self._ctrl[i][time][0] for i in range(self._num_actions)]),
                                                                         np.array([self._delta_ctrl[i][time][0] for i in range(self._num_actions)]))

            for controls in range(self._num_actions):
                self._delta_ctrl[controls][-1][sample] = np.random.uniform(-1, 1, 1)[0] * self._variance

    def simulate(self, viewer=None):
        for iteration in range(self._time_horizon):

            self.compute_controls()
            self._cost_avg[iteration] = np.sum(self._sample_cost)
            for time in range(self._horizon):
                entropy = self.total_entropy([self._delta_ctrl[i][time][:] for i in range(self._num_actions)])

                for controls in range(self._num_actions):
                    self._ctrl[controls][time][0] += entropy[controls]

            for controls in range(self._num_actions):
                print(self._ctrl[controls][0][0])
                self._plant.data.ctrl[controls] = self._ctrl[controls][0][0]

            self.plant_control[iteration] = np.array([self._ctrl[i][0][0] for i in range(self._num_actions)])

            self._plant.step()
            if viewer is not None:
                viewer.render()

            self._plant_state[iteration] = self._plant.get_state()

            for controls in range(self._num_actions):
                np.roll(self._ctrl[controls], -1)
                self._ctrl[controls][-1][0] = 0

            if iteration % 25 == 0:
                print(iteration)


if __name__ == "__main__":
    model = load_model_from_path("xmls/finger.xml")
    sim = MjSim(model)
    plant = MjSim(model)
    new_state = State(time=0, qpos=np.array([-np.pi/2 + 0.2, 0, 0+0.1]), qvel=np.array([0.0, 0, 0]), act=0, udd_state={})
    plant.set_state(new_state)

    Params = {"R":  700.0, "Lambda": 25000, "Samples": 500, "Horizon": 700, "Time": 2500, "Variance": 0.9}

    cost = Cost(None, np.eye(2) * Params["R"], None, Params["Lambda"])
    pi = MPPI(sim, Params["Samples"], Params["Horizon"], cost, Params["Variance"], plant, Params["Time"])
    pi.simulate()

    np.save(
        f'mppi/results/working_controls_finger_{Params["R"]}_'
        f'                                     {Params["Lamda"]}_'
        f'                                     {Params["Samples"]}_'
        f'                                     {Params["Horizon"]}_'
        f'                                     {Params["Time"]}_'
        f'                                     {Params["variance"]}.npy',
                                               pi.plant_control[:]
    )

    rec = input("Visualise ?")
    print("Visualising")
    viewer = MjViewer(plant)

    plant.set_state(new_state)

    for control in range(len(pi.plant_control)):
        plant.data.ctrl[0] = pi.plant_control[control][0]
        plant.data.ctrl[1] = pi.plant_control[control][1]
        plant.step()
        viewer.render()
