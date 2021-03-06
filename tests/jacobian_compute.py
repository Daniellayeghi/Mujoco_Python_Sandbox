from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
from collections import namedtuple
from scipy.optimize import approx_fprime
from utils.utilities import State

# This module calculates the derivatives of dynamics w.r.t states and controls fprime.
# The main reason for this module is to be used in unit tests for C++ implementations

def forward_sim_state(state_vector: np.array):
    perturbed_state = sim.get_state()
    row = state_vector.shape[0]
    for joint in range(perturbed_state.qpos.shape[0]):
        perturbed_state.qpos[joint] = state_vector[joint]

    for joint in range(perturbed_state.qvel.shape[0]):
        perturbed_state.qvel[joint] = state_vector[joint+int(row/2)]

    sim.set_state(perturbed_state)
    sim.step()

    result_pos = np.array([sim.data.qpos[i] for i in range(sim.data.qpos.shape[0])]).reshape(sim.data.qpos.shape[0], 1)
    result_vel = np.array([sim.data.qvel[i] for i in range(sim.data.qvel.shape[0])]).reshape(sim.data.qvel.shape[0], 1)

    return np.vstack((result_pos, result_vel))


def forward_sim_acc_state(state_vector: np.array):
    perturbed_state = sim.get_state()
    row = state_vector.shape[0]
    for joint in range(perturbed_state.qpos.shape[0]):
        perturbed_state.qpos[joint] = state_vector[joint]

    for joint in range(perturbed_state.qvel.shape[0]):
        perturbed_state.qvel[joint] = state_vector[joint+int(row/2)]

    sim.set_state(perturbed_state)
    sim.step()

    result = np.array([sim.data.qacc[i] for i in range(sim.data.qacc.shape[0])])

    return result


def forward_sim_ctrl(ctrl_vector: np.array, state: State):
    sim.set_state(state)
    row = state.qpos.shape[0]
    result = np.zeros(state.qpos.shape[0] + state.qvel.shape[0])
    for joint in range(sim.data.ctrl.shape[0]):
        sim.data.ctrl[joint] = ctrl_vector[joint]

    sim.step()

    for joint in range(state.qpos.shape[0]):
        result[joint] = sim.data.qpos[joint]

    for joint in range(state.qvel.shape[0]):
        result[joint + row] = sim.data.qvel[joint]

    return result


def forward_sim_ctrl_acc(ctrl_vector: np.array, state: State):
    sim.set_state(state)
    for joint in range(sim.data.ctrl.shape[0]):
        sim.data.ctrl[joint] = ctrl_vector[joint]

    sim.step()

    result = np.array([sim.data.qacc[i] for i in range(sim.data.qacc.shape[0])])

    return result


if __name__ == "__main__":
    model = load_model_from_path("/home/daniel/Repos/OptimisationBasedControl/models/cartpole.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)

    new_state = State(time=0, qpos=np.array([0, np.pi]), qvel=np.array([0, 0]), act=0, udd_state={})
    sim.set_state(new_state)

    state_vec = np.array([new_state.qpos[0], new_state.qpos[1], new_state.qvel[0], new_state.qvel[1]])

    epsilon = 0.000001
    # sim.set_state(new_state)
    # J_state = np.vstack(
    #     [approx_fprime(state_vec, lambda x: forward_sim_state(x)[m], epsilon)
    #      for m in range(state_vec.shape[0])]
    # )
    #
    # print(J_state)

    sim.set_state(new_state)
    ctrl_vec = np.array([0, 0])
    J_ctrl = np.vstack(
        [approx_fprime(ctrl_vec, lambda x: forward_sim_ctrl(x, new_state)[m], epsilon)
         for m in range(state_vec.shape[0])]
    )

    print(J_ctrl)

    # sim.set_state(new_state)
    # ctrl_vec = np.array([0.5, 0.3])
    # J_ctrl_acc = np.vstack(
    #     [approx_fprime(ctrl_vec, lambda x: forward_sim_ctrl_acc(x, new_state)[m], epsilon)
    #      for m in range(sim.data.qacc.shape[0])]
    # )

    # print(J_ctrl_acc)
    #
    # sim.set_state(new_state)
    # state_vec = np.array([new_state.qpos[0], new_state.qpos[1], new_state.qvel[0], new_state.qvel[1]])
    # J_state_acc = np.vstack(
    #     [approx_fprime(state_vec, lambda x: forward_sim_acc_state(x)[m], epsilon)
    #      for m in range(sim.data.qacc.shape[0])]
    # )
