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
        self._values = np.zeros([states.shape[1], states.shape[1]])

    def solve_disc_value_iteration(self, sim: MjSim, cost_func):
        [row, col] = np.shape(states)
        for iter in range(5):
            print(f"iteration: {iter}")
            for s_1 in range(col):
                for s_2 in range(col):
                    for ctrl in range(self._ctrls.shape[0]):
                        # reinitialise the state
                        sim.set_state(
                            State(time=0, qpos=np.array([states[0][s_1]]), qvel=np.array([states[1][s_2]]), act=0,
                                  udd_state={})
                        )
                        sim.data.ctrl[0] = self._ctrls[ctrl]
                        sim.step()
                        # solve for the instantaneous cost and interpolate the value at the next state
                        value_curr = cost_func(
                            np.append(sim.data.xipos[1], sim.data.qvel[0]), sim.data.ctrl
                        ) + interpolate.interp2d(
                            self._states[0][:], self._states[1][:], self._values, kind='linear'
                        )(sim.data.qpos[0], sim.data.qvel[0])
                        # Hacky fix for the first iteration of vi
                        if ctrl == 0 and iter == 0:
                            self._values[s_1][s_2] = value_curr
                        if value_curr < self._values[s_1][s_2]:
                            self._values[s_1][s_2] = value_curr[0]
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


