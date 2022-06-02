
import time
from collections import namedtuple

import numpy as np
import torch
from mujoco_py import load_model_from_path, MjSim, MjViewer
from scipy.optimize import approx_fprime
from torch.autograd import Variable

State = namedtuple('State', 'time qpos qvel act udd_state')
np.set_printoptions(threshold=np.inf)
torch.manual_seed(1)  # reproducible

# Assume the state: [[x1, x2, ... ], [xd1, xd2, ...]]
def dynamics_model(sim, state: State, ctrl: np.ndarray):
    sim.set_state(state)
    sim.data.ctrl[0] = ctrl
    sim.step()

    # Stack state to result
    result = np.zeros(sim.data.qpos.shape[0] + sim.data.qvel.shape[0])
    result[0:sim.data.qpos.shape[0]] = sim.data.qpos
    result[sim.data.qpos.shape[0]:sim.data.qpos.shape[0] + sim.data.qvel.shape[0]] = sim.data.qvel

    return result


class HJBSolver(object):
    def __init__(self, model_file: str, simulator, state_ctrl_size: tuple):
        self.value_function = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

        self.value_function.load_state_dict(torch.load("state_dict_model.pt"))
        self.value_function.eval()

        self._input_states = torch.empty(state_ctrl_size[0], 1)
        self._output_value = torch.zeros(1, 1)

        self._input_states_var = Variable(self._input_states)
        self._output_value_var = Variable(self._output_value)
        self._input_states_var.requires_grad = True
        self._output_value_var.requires_grad = True
        self.simul = simulator
        self.ctrl_vec = np.zeros_like(simulator.data.ctrl)

    def compute_ctrl(self, state: np.array):
        self._input_states = torch.from_numpy(state)
        self._input_states_var = Variable(self._input_states)
        self._input_states_var.requires_grad = True

        current_value = self.value_function(self._input_states_var.float())
        print(f"Value: {current_value} for {state}")
        dv_dx = torch.autograd.grad(current_value, self._input_states_var, retain_graph=True)[0]

        current_state = State(
            time=0,
            qpos=np.array([state[0]]),
            qvel=np.array([state[1]]),
            act=0,
            udd_state={}
        )

        j_ctrl = np.vstack(
            [approx_fprime(
                self.ctrl_vec, lambda x: dynamics_model(self.simul, current_state, x)[m], 1e-6)
                for m in range(current_state.qpos.shape[0] + current_state.qvel.shape[0])]
        )

        self.ctrl_vec = -0.5 * 1 * (dv_dx.numpy().dot(j_ctrl))

        return self.ctrl_vec


if __name__ == "__main__":
    model = load_model_from_path("../xmls/doubleintegrator.xml")
    sim = MjSim(model)
    sim_cp = MjSim(model)

    viewer = MjViewer(sim)
    state = State(time=0, qpos=np.array([1]), qvel=np.array([0]), act=0, udd_state={})
    sim.set_state(state)

    hjb_solver = HJBSolver("state_dict_model.pt", sim_cp, (2, 1))

    while True:
        viewer.render()
        sim.step()
        ctrl = hjb_solver.compute_ctrl(np.array([sim.data.qpos[0], sim.data.qvel[0]]))
        sim.data.ctrl[0] = ctrl[0]
        time.sleep(0.01)


