from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from mujoco_py import load_model_from_path, MjSim
from scipy.optimize import approx_fprime
from torch.autograd import Variable

State = namedtuple('State', 'time qpos qvel act udd_state')
np.set_printoptions(threshold=np.inf)
torch.manual_seed(1)  # reproducible


# Assume the state: [[x1, x2, ... ], [xd1, xd2, ...]]
def dynamics_model(sim, state: State, ctrl: np.ndarray):
    sim.set_state(state)
    sim.data.ctrl[0] = ctrl[0]
    sim.step()

    # Stack state to result
    result = np.zeros(sim.data.qpos.shape[0] + sim.data.qvel.shape[0])
    result[0:sim.data.qpos.shape[0]] = sim.data.qpos
    result[sim.data.qpos.shape[0]:sim.data.qpos.shape[0] + sim.data.qvel.shape[0]] = sim.data.qvel

    return result


class QRCost(object):
    def __init__(self, q: np.array, r: np.array, ref: np.array):
        self._Q = q
        self._R = r
        self._ref = ref
        self._state_cost = 0
        self._ctrl_cost = 0

    def cost_function_state(self, state):
        err = self._ref - state
        self._state_cost = err.dot(self._Q).dot(err)

    def cost_function_ctrl(self, ctrl):
        self._ctrl_cost = ctrl.dot(self._R).dot(ctrl)

    def running_cost(self):
        return self._state_cost + self._ctrl_cost


class ValueGradient(object):
    BATCH_SIZE = 64
    EPOCH = 20

    def __init__(self, states: np.array, ctrls: np.array):
        try:
            # Assumes state as 2 dimensional
            assert(states.shape[0] <= 2)
        except AssertionError as error:
            print(error)

        self._states = states
        self._ctrls  = ctrls

        # Network parameters
        self._input_states = torch.empty(states.shape[0], 1)
        self._output_value = torch.zeros(1, 1)
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(2, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, 1),
        )

        self._optimizer = torch.optim.Adam(self.value_net.parameters(), lr=0.0001)
        self._loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

        # Variable setup for autodiff
        self._input_states_var = Variable(self._input_states)
        self._output_value_var = Variable(self._output_value)
        self._input_states_var.requires_grad = True
        self._output_value_var.requires_grad = True

    def solve_disc_value_iteration(self, model: MjSim, cost_func):
        [row, col] = np.shape(states)
        eps = 0.5
        ctrl_vec = np.zeros_like(sim.data.ctrl)
        mdl = load_model_from_path("../xmls/Pendulum.xml")
        sim_2 = MjSim(mdl)
        for iter in range(10):
            print(f"Iteration: {iter}")
            for s_1 in range(col):
                for s_2 in range(col):
                    for epoch in range(100):
                        # reinitialise the state
                        state = State(
                            time=0,
                            qpos=np.array([states[0][s_1]]),
                            qvel=np.array([states[1][s_2]]),
                            act=0 ,
                            udd_state={}
                        )

                        model.set_state(state)

                        j_ctrl = np.vstack(
                            [approx_fprime(
                                ctrl_vec, lambda x: dynamics_model(sim_2, state, x)[m], eps)
                                for m in range(state.qpos.shape[0] + state.qvel.shape[0])]
                        )

                        cost_func.cost_function_state(
                            np.append(model.data.qpos[0], model.data.qvel[0])
                        )

                        self._input_states = torch.from_numpy(np.array([model.data.qpos, model.data.qvel]).T)
                        self._input_states_var = Variable(self._input_states)
                        self._input_states_var.requires_grad = True

                        current_value = self.value_net(self._input_states_var.float())
                        dv_dx = torch.autograd.grad(current_value, self._input_states_var, retain_graph=True)[0]
                        ctrl_vec = -0.5*1*(dv_dx.numpy().dot(j_ctrl))[0]

                        model.data.ctrl[0] = ctrl_vec[0]
                        cost_func.cost_function_ctrl(model.data.ctrl)
                        model.step()

                        self._input_states = torch.from_numpy(np.array([model.data.qpos, model.data.qvel]).T)
                        self._input_states_var = Variable(self._input_states)
                        self._input_states_var.requires_grad = True

                        future_value = self.value_net(self._input_states_var.float())
                        self._output_value_var = Variable(cost_func.running_cost() + future_value)
                        self._output_value_var.requires_grad = True
                        loss = self._loss_func(current_value, self._output_value_var)
                        self._optimizer.zero_grad()  # clear gradients for next train
                        loss.backward()
                        self._optimizer.step()

                        if epoch % 25 == 0:
                            print(f"Loss is: {loss}")
                            # print(f"Jacobian: {j_ctrl}")
                            # print(f"Ctrl: {ctrl_vec}")
                            # print(f"dv_dx: {dv_dx.numpy()}")


if __name__ == "__main__":
    # Setup quadratic cost
    cost = QRCost(
        np.diagflat(np.array([500, 500 * 0.05])),
        np.diagflat(np.array([1])),
        np.array([np.pi, 0])
    )

    # Setup value iteration
    disc_state = 20
    disc_ctrl = 50
    pos_arr = (np.linspace(-np.pi, np.pi*3, disc_state))
    vel_arr = (np.linspace(-np.pi, np.pi, disc_state))
    states = np.array((pos_arr, vel_arr))
    ctrls = np.linspace(-1, 1, disc_ctrl)
    vg = ValueGradient(states, ctrls)

    # Load environment
    mdl = load_model_from_path("../xmls/doubleintegrator.xml")
    sim = MjSim(mdl)

    # Solve value iteration
    vg.solve_disc_value_iteration(sim, cost)
    pos_tensor = torch.from_numpy(np.linspace(-np.pi, np.pi, disc_state))
    vel_tensor = torch.from_numpy(np.linspace(-np.pi, np.pi, disc_state))
    prediction = np.zeros((disc_state, disc_state))

    example = torch.zeros(1, 2)
    traced_script_module = torch.jit.trace(vg.value_net, example)
    traced_script_module.save("value_net_hjb.pt")

    for pos in range(pos_tensor.numpy().shape[0]):
        for vel in range(vel_tensor.numpy().shape[0]):
            prediction[pos][vel] = vg.value_net(
                torch.from_numpy(np.array([pos_tensor[pos], vel_tensor[vel]])
                                 ).float()).detach().numpy()[0]

    min = np.unravel_index(prediction.argmin(), prediction.shape)
    print(f"The min value is at pos {pos_tensor.numpy()[min[0]]} and vel {vel_tensor.numpy()[min[1]]}")

    print(prediction)
    # Plot the structure of cost to go
    [P, V] = np.meshgrid(pos_arr, vel_arr)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(P, V, prediction, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('surface')
    ax.set_xlabel('Pos')
    ax.set_ylabel('Vel')
    ax.set_zlabel('Value')
    plt.show()
