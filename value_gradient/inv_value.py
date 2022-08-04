from mlp_torch import MLP
from net_utils_torch import LayerInfo
from collections import namedtuple
import pandas as pd
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
# import mujoco
# from mujoco import derivative

# Load Data
parent_path = "../../OptimisationBasedControl/data/"
data = pd.read_csv(parent_path + "di_data_value.csv", sep=',', header=None).to_numpy()
n_traj, n_train = 5000, int(5000 * 0.75)

d_train = data[0:n_train, :]
d_test = data[n_train:, :]
d_train_d = TensorDataset(torch.Tensor(d_train))
d_loader = DataLoader(d_train_d, batch_size=64, shuffle=True)
DataParams = namedtuple('DataParams', 'n_state, n_pos, n_vel, n_ctrl, n_desc, n_batch')
d_params = DataParams(2, 1, 1, 1, 2, d_loader.batch_size)

# Value network
val_input, value_output = d_params.n_state + d_params.n_desc, d_params.n_ctrl
v_layers = [[val_input, 16, value_output], [], 0]
value_net = MLP(LayerInfo(*v_layers))

# Policy network
policy_input, policy_output = d_params.n_state + d_params.n_desc, d_params.n_pos
p_layers = [[val_input, 16, 16, value_output], [], 0]
policy_net = MLP(LayerInfo(*p_layers))


class ValueFunction(MLP):
    def __init__(self, data_params: DataParams, layer_info: LayerInfo, apply_sigmoid=False):
        super(ValueFunction, self).__init__(layer_info, apply_sigmoid)
        self.loss = list()
        self._data_params = data_params
    def compute_loss(self, x_curr, x_next, x_goal):
        self.loss.append()

    def goal_loss(self, x_goal):
        return 10 * self.forward(x_goal)

    def descent_loss(self, x_curr, x_next, u_curr):



class OptimalPolicy(torch.nn.Module):
    def __init__(
            self, policy: MLP, value: MLP, params: DataParams,
            derivative=lambda x1, x2, dt=.01: (x2 - x1) / dt,
            integrate=lambda x1, x2, dt=0.01: x1 + x2 * dt
    ):
        super(OptimalPolicy, self).__init__()
        self.policy = policy
        self.value = value
        self.derivative = derivative
        self.integrate = integrate
        self.params = params
        self.final_pos = torch.zeros((self.params.n_pos, self.params.n_batch))
        self.final_vel = torch.zeros((self.params.n_vel, self.params.n_batch))
        self.final_acc = torch.zeros((self.params.n_vel, self.params.n_batch))
        self.final_state = torch.zeros((self.params.n_vel * 2, self.params.n_batch))

    # Policy net need to output positions. From there derivatives can compute velocity.
    # Projection onto value function computes new acceleration, then integrate to get state
    def forward(self, inputs):
        pos, vel = inputs[0:self.params.n_pos], inputs[self.params.n_pos:self.params.n_vel + self.params.n_pos]
        self.final_pos = self.policy(inputs)
        value = self.value(inputs)
        self.final_state[0, self.params.n_vel, :] = self.derivative(pos, self.final_pos)
        self.final_state[self.params.n_vel, 2*self.params.n_vel] = self.derivative(
            vel, self.self.final_state[0, self.params.n_vel, :]
        )

        # Compute network derivative w.r.t state
        d_value = torch.autograd.grad(
            [a for a in value], [inputs[0:inputs.size()[0]-self.params.n_desc]], create_graph=True, only_inputs=True
        )[0]

        # Return the new state form
        self.final_state = self.final_state - d_value * (F.relu(
            (d_value*self.final_state).sum(dim=1)
        )/(d_value**2).sum(dim=1))[:, None]

        self.final_acc = self.final_state[self.params.n_vel:, :]
        self.final_vel = self.integrate(vel, self.final_state[self.params.n_vel:, :])
        self.final_pos = self.integrate(pos, self.final_vel)

        return self.final_pos, self.final_vel, self.final_acc


# Convert to dataset
d_train_d = TensorDataset(torch.Tensor(d_train.transpose()))
d_loader = DataLoader(d_train_d, batch_size=64, shuffle=True)
lr = 1e-4
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, policy_net.parameters()), lr=lr)
