
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from mujoco_py import load_model_from_path, MjSim
from scipy import interpolate

State = namedtuple('State', 'time qpos qvel act udd_state')
np.set_printoptions(threshold=np.inf)


class QRCost(object):
    def __init__(self, q: np.array, r: np.array, ref: np.array):
        self._Q = q
        self._R = r
        self._ref = ref

    def cost_function(self, state, ctrl: np.array):
        err = self._ref - state
        state_cost = err.dot(self._Q).dot(err)
        ctrl_reg   = ctrl.dot(self._R).dot(ctrl)
        return state_cost + ctrl_reg


class ValueGradient(object):
    def __init__(self, states: np.array, ctrls: np.array):
        try:
            # Assumes state as 2 dimensional
            assert(states.shape[0] <= 2)
        except AssertionError as error:
            print(error)

        self._states = states
        self._ctrls  = ctrls
        self.P, self.V = np.meshgrid(states[0][:], states[1][:])
        self._values = np.zeros([states.shape[1], states.shape[1]])

    def solve_disc_value_iteration(self, model: MjSim, cost_func):
        [row, col] = np.shape(states)
        for iter in range(2):
            print(f"iteration: {iter}")
            for s_1 in range(col):
                for s_2 in range(col):
                    for ctrl in range(self._ctrls.shape[0]):
                        # reinitialise the state
                        model.set_state(
                            State(
                                time=0,
                                qpos=np.array([states[0][s_1]]),
                                qvel=np.array([states[1][s_2]]),
                                act=0,
                                udd_state={}
                            )
                        )
                        model.data.ctrl[0] = self._ctrls[ctrl]
                        model.step()
                        # solve for the instantaneous cost and interpolate the value at the next state
                        value_curr = cost_func(
                            np.append(model.data.xipos[1], model.data.qvel[0]), model.data.ctrl
                        )

                        if (self._states[0][0] < model.data.qpos[0] < self._states[0][-1]) and \
                                self._states[1][0] < model.data.qvel[0] < self._states[1][-1]:
                            value_curr += interpolate.griddata(
                                np.array([self.P.ravel(), self.V.ravel()]).T, self._values.ravel(),
                                np.array([model.data.qpos[0], model.data.qvel[0]]).T
                            )
                        else:
                            rbf_net = interpolate.Rbf(
                                self.P.ravel(), self.V.ravel(), self._values.ravel(), function="linear"
                            )
                            value_curr += rbf_net(model.data.qpos[0], model.data.qvel[0])

                        # Hacky fix for the first iteration of vi
                        if ctrl == 0 and iter == 0:
                            if np.isnan(value_curr):
                                value_curr = 800
                            self._values[s_1][s_2] = value_curr
                        if value_curr < self._values[s_1][s_2]:
                            # print(f"Difference = {value_curr - self._values[s_1][s_2]} index {s_1}{s_2}")
                            self._values[s_1][s_2] = value_curr
        return self._values


if __name__ == "__main__":
    # Setup quadratic cost
    cost = QRCost(
        np.diagflat(np.array([500, 0, 500, 500 * 0.05])),
        np.diagflat(np.array([1])),
        np.array([1.46152155e-17, 0.00000000e+00, 1.19342291e-01, 0])
    )

    # Setup value iteration
    disc_state = 25
    disc_ctrl = 25
    pos_arr = (np.linspace(-np.pi, np.pi*3, disc_state))
    vel_arr = (np.linspace(-np.pi, np.pi, disc_state))
    states = np.array((pos_arr, vel_arr))
    ctrls = np.linspace(-1, 1, disc_ctrl)
    vg = ValueGradient(states, ctrls, )

    # Load environment
    mdl = load_model_from_path("../xmls/Pendulum.xml")
    sim = MjSim(mdl)

    # Solve value iteration
    values = vg.solve_disc_value_iteration(sim, cost.cost_function)
    min = np.unravel_index(values.argmin(), values.shape)
    print(f"The min value is at pos {pos_arr[min[0]]} and vel {vel_arr[min[1]]}")

    # Plot the structure of cost to go
    [P, V] = np.meshgrid(pos_arr, vel_arr)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(P, V, values, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('surface')
    ax.set_xlabel('Pos')
    ax.set_ylabel('Vel')
    ax.set_zlabel('Value')
    plt.show()
