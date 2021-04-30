
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from mujoco_py import load_model_from_path, MjSim
from scipy import interpolate
import torch
from torch.autograd import Variable
import torch.utils.data as Data

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
        for iter in range(6):
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
                        value_curr = cost_func(
                            np.append(model.data.xipos[1], model.data.qvel[0]), model.data.ctrl
                        )
                        model.step()
                        # solve for the instantaneous cost and interpolate the value at the next state

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
                            self._values[s_1][s_2] = value_curr
                        if value_curr < self._values[s_1][s_2]:
                            # print(f"Difference = {value_curr - self._values[s_1][s_2]} index {s_1}{s_2}")
                            self._values[s_1][s_2] = value_curr
        return self._values


class FittedValueIteration(object):
    def __init__(self, output_val, input_state):
        # Network parameters
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(2, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 240),
            torch.nn.ReLU(),
            torch.nn.Linear(240, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 1),
        )

        self._optimizer = torch.optim.Adam(self.value_net.parameters(), lr=0.01)
        self._loss_func = torch.nn.MSELoss()  # this is for regression mean squared los

        # Generate tensors from data
        self._output_val_t = torch.from_numpy(output_val)
        self._input_state_t = torch.from_numpy(input_state)
        self._output_val_t = Variable(self._output_val_t)
        self._input_state_t = Variable(self._input_state_t)

        # Create dataset for batch
        self._torch_data_set = Data.TensorDataset(self._input_state_t, self._output_val_t)
        self._loader = Data.DataLoader(
            dataset=self._torch_data_set,
            batch_size=20,
            shuffle=True,
            num_workers=5,
        )

    def batch_train_net(self):
        for epoch in range(800):
            for step, (batch_in, batch_out) in enumerate(self._loader):
                # Compute net prediction and loss:
                pred = self.value_net(Variable(batch_in).float())
                loss = self._loss_func(pred, Variable(batch_out).float())

                # Backprop through the net
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if epoch % 10 == 0:
                    print(f"Loss: {loss}")


if __name__ == "__main__":
    # Setup quadratic cost
    cost = QRCost(
        np.diagflat(np.array([100, 0, 5000, 10 * 0.01])),
        np.diagflat(np.array([1])),
        np.array([1.46152155e-17, 0.00000000e+00, 1.19342291e-01, 0])
    )

    # Setup value iteration
    disc_state = 20
    disc_ctrl = 10
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
    min_val = np.unravel_index(values.argmin(), values.shape)
    print(f"The min value is at pos {pos_arr[min_val[0]]} and vel {vel_arr[min_val[1]]}")


    [P, V] = np.meshgrid(pos_arr, vel_arr)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(P, V, values, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('surface')
    ax.set_xlabel('Pos')
    ax.set_ylabel('Vel')
    plt.show()
    ax.set_zlabel('Value')

    # Fit value grid to nn
    v_r = values.ravel()
    ftv = FittedValueIteration(np.reshape(v_r, (v_r.shape[0], 1)), np.vstack([P.ravel(), V.ravel()]).T)
    # ftv.fit_network(values, np.vstack([P.ravel, vel_arr]).T)
    ftv.batch_train_net()
    torch.save(ftv.value_net.state_dict(), "state_dict_model.pt")

    # Export model for c++
    example = torch.zeros(1, 2)
    traced_script_module = torch.jit.trace(ftv.value_net, example)
    traced_script_module.save("traced_value_model.pt")

    pos_tensor = torch.from_numpy(pos_arr)
    vel_tensor = torch.from_numpy(vel_arr)
    prediction = np.zeros((disc_state, disc_state))

    for pos in range(pos_tensor.numpy().shape[0]):
        for vel in range(vel_tensor.numpy().shape[0]):
            prediction[vel][pos] = ftv.value_net(
                torch.from_numpy(np.array([pos_tensor[pos], vel_tensor[vel]])
                                 ).float()).detach().numpy()[0]

    # Plot the structure of cost to go
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(P, V, prediction, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax1.set_xlabel('Pos')
    ax1.set_ylabel('Vel')
    ax1.set_zlabel('Value')
    ax1.set_title("Approximation")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(P, V, values, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax2.set_xlabel('Pos')
    ax2.set_ylabel('Vel')
    ax2.set_zlabel('Value')
    ax2.set_title("Target")
    plt.show()

