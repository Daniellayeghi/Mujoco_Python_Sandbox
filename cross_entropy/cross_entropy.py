from collections import namedtuple
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
from numpy.random import multivariate_normal
from collections import namedtuple

State = namedtuple('State', 'time qpos qvel act udd_state')
CEMParams = namedtuple('CEMParams', 'n_samples n_time mean cov')


def set_control(ctrl, new_ctrl):
    for row in range(ctrl.shape[0]):
        ctrl[row] = new_ctrl[row]


class CostFunction(object):
    def __init__(self, running_cost, terminal_cost, state_goal, ctrl_goal):
        self._state_goal = state_goal
        self._ctrl_goal = ctrl_goal
        self.running_cost = lambda state, ctrl: running_cost(self._state_goal - state, self._ctrl_goal - ctrl)
        self.terminal_cost = lambda state: terminal_cost(self._state_goal - state)


class CrossEntropy(object):
    def __init__(self, model, cem_params: CEMParams):
        self._model = model
        self._sim = MjSim(model)
        self._samples, self._time, self._mean, self._covariance = cem_params
        self._delta_ctrl_samples = np.zeros((self._samples, self._time * model.nu))
        self._ctrl_samples = np.zeros(self._time * model.nu)
        self._ctrl_mean = np.zeros(self._time * model.nu)
        self._ctrl_var = np.zeros((model.nu, self._time * model.nu))
        self._sample_cost = np.zeros(self._samples)

    def sample_controls(self):
        for sample in range(self._samples):
            self._delta_ctrl_samples[sample, :] = multivariate_normal(self._mean, self._covariance, self._time).T

    def evaluate_traj_cost(self, intial_state, cost:CostFunction):
        for sample in range(self._samples):
            self._sim.set_state(intial_state)
            for time in range(self._time):
                beg = time * self._model.nu
                end = beg + self._model.nu
                set_control(self._sim.data.ctrl, self._ctrl_samples[beg:end] + self._delta_ctrl_samples[sample][beg:end])
                self._sim.step()
                state = self._sim.get_state()
                self._sample_cost[sample] += cost.running_cost(np.concatenate((state.qpos, state.qvel)), self._sim.data.ctrl)
            state = self._sim.get_state()
            self._sample_cost[sample] += cost.terminal_cost(np.concatenate((state.qpos, state.qvel)))

    def compute_mean_cov(self):
        sort_index = np.argsort(self._sample_cost)
        for time in range(self._time):
            beg = time * self._model.nu
            end = beg + self._model.nu
            for sample in sort_index[0:self._samples - int(self._samples*0.2)]:
                ctrl_sample = self._delta_ctrl_samples[sample][beg:end]
                self._ctrl_mean[beg:end] += (ctrl_sample/(self._samples - int(self._samples*0.2)))
                self._ctrl_var[:, beg:end] += np.outer(
                    (ctrl_sample - self._mean), (ctrl_sample - self._mean)
                )/(self._samples - int(self._samples*0.2))

            self._ctrl_samples[beg:end] += self._ctrl_mean[beg:end]

    def average_mean_cov(self):
        average_mean = np.zeros(model.nu)
        average_cov = np.zeros((model.nu, model.nu))
        denomenator = 0
        for time in range(self._time):
            beg = time * self._model.nu
            end = beg + self._model.nu
            average_mean += ((self._time - time) * self._ctrl_mean[beg:end])
            average_cov += ((self._time - time) * self._ctrl_var[:, beg:end])
            denomenator += (self._time - time)

        self._mean = average_mean/denomenator
        self._covariance = average_cov/denomenator
        print(self._mean, self._covariance)

    def control(self, initial_state, cost_function):
        self._ctrl_mean.fill(0)
        self._ctrl_var.fill(0)
        self.sample_controls()
        self.evaluate_traj_cost(initial_state, cost_function)
        self.compute_mean_cov()
        self.average_mean_cov()
        control = self._ctrl_samples[0:model.nu]
        self._ctrl_samples = np.roll(self._ctrl_samples, -model.nu)
        self._ctrl_samples[-model.nu:] = np.zeros(model.nu)
        return control


if __name__ == "__main__":
    model = load_model_from_path("../xmls/cartpole.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)
    Q = np.eye(model.nq + model.nv)
    np.fill_diagonal(Q, np.array([0, 0, 0.001, 0.001]))
    Q_terminal = np.eye(model.nq + model.nv)
    np.fill_diagonal(Q_terminal, np.array([1000000, 1000000, 10000, 10000]))
    R = np.eye(model.nu)
    np.fill_diagonal(R, np.array([0.001]))
    start_goal = np.array([0, 0, 0, 0])
    ctrl_goal = np.array([1])

    cost_function = CostFunction(lambda state_err, ctrl_err:state_err.dot(Q).dot(state_err) + ctrl_err.dot(R).dot(ctrl_err),
                                 lambda state_err:state_err.dot(Q_terminal).dot(state_err),start_goal,ctrl_goal)
    cem_params = CEMParams(20, 100, np.zeros(model.nu), np.eye(model.nu)*0.5)
    cem_control = CrossEntropy(model, cem_params)

    sim.set_state(State(time=0, qpos=np.array([0, 0]), qvel=np.array([0, 0]), act=0, udd_state={}))

    while True:
        ctrl = cem_control.control(sim.get_state(), cost_function)
        print(ctrl)
        for u in range(model.nu):
            sim.data.ctrl[u] = ctrl[u]
        sim.step()
        viewer.render()



