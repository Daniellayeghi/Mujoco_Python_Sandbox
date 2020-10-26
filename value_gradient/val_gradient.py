from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple

State = namedtuple('State', 'time qpos qvel act udd_state')


class QRCost(object):
    def __init__(self, q: np.array, r: np.array):
        self._Q = q
        self._R = r
        self._ref = np.array([1.46152155e-17, 0.00000000e+00, 1.19342291e-01])

    def cost_function(self, state, ctrl: np.array):
        err = -self._ref + state[1]
        state_cost = err.dot(self._Q).dot(err)
        ctrl_reg   = ctrl.dot(self._R).dot(ctrl)
        return state_cost + ctrl_reg


class ValueGradient(object):
    def __init__(self, state_size: int, cost_function, discrete_num: int):
        try:
            # Assumes state as 2 dimensional
            assert(state_size <= 2)
        except AssertionError as error:
            print(error)

        self._cost_func = cost_function
        self._values    = np.ones([discrete_num, discrete_num]) * -np.inf

    def solve_disc_value_iteration(self, model: MjSim, states, ctrls):
        [row, col] = np.shape(states)
        for s_1 in range(col):
            for s_2 in range(col):
                for ctrl in range(ctrls.shape[0]):
                    model.set_state(
                        State(time=0, qpos=np.array([states[0][s_1]]), qvel=np.array([states[1][s_2]]), act=0,
                              udd_state={})
                    )
                    model.data.ctrl[0] = ctrls[ctrl]
                    model.step()
                    cost = self._cost_func(model.data.xipos, model.data.ctrl)
                    if cost > self._values[s_1][s_2]:
                        self._values[s_1][s_2] = cost

        return self._values


if __name__ == "__main__":
    cost = QRCost(np.diagflat(np.array([100, 100, 100 * 0.01])), np.diagflat(np.array([10])))
    disc_state = 100
    disc_ctrl  = 200
    pos_arr = (np.linspace(-5, 5, disc_state))
    vel_arr = (np.linspace(-5, 5, disc_state))
    states  = np.array((pos_arr, vel_arr))
    vg = ValueGradient(2, cost.cost_function, disc_ctrl)
    mdl = load_model_from_path("/home/daniel/Repos/mujoco_py_test/xmls/Pendulum.xml")
    sim = MjSim(mdl)
    values = vg.solve_disc_value_iteration(sim, states, np.linspace(-1, 1, disc_ctrl))
    [P, V] = np.meshgrid(pos_arr, vel_arr)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(P, V, values, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('surface')
    plt.show()


