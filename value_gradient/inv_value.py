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
parent_path = "../../sr3-matlab/demo/"
values = pd.read_csv(parent_path + "di_values.csv", sep=',', header=None).to_numpy()
states = pd.read_csv(parent_path + "di_states.csv", sep=',', header=None).to_numpy()
desc = pd.read_csv(parent_path + "di_conds.csv", sep=',', header=None).to_numpy()  # init; goal
n_traj, n_train = 5000, int(5000 * 0.75)
input_q = np.vstack((states, values, desc))
d_train = input_q[:, 0:n_train]
d_test = input_q[:, n_train:]
d_train_d = TensorDataset(torch.Tensor(d_train.transpose()))
d_loader = DataLoader(d_train_d, batch_size=64, shuffle=True)
DataParams = namedtuple('DataParams', 'n_state, n_pos, n_vel, n_ctrl, n_desc, n_batch')
d_params = DataParams(2, 1, 1, 1, desc.size, d_loader.batch_size)

# Value network
val_input, value_output = d_params.n_state + d_params.n_desc, d_params.n_ctrl
v_layers = [[val_input, 16, value_output], [], 0]
value_net = MLP(LayerInfo(*v_layers))

# Policy network
policy_input, policy_output = d_params.n_state + d_params.n_desc, d_params.n_pos
p_layers = [[val_input, 16, 16, value_output], [], 0]
policy_net = MLP(LayerInfo(*p_layers))


class OptimalPolicy(torch.nn.Module):
    def __init__(self, policy: MLP, value: MLP, n_params: DataParams, integrate=lambda x1, x2, delta=.01: x1 + x2 * delta):
        super(OptimalPolicy, self).__init__()
        self.policy = policy
        self.value = value
        self.derivative = integrate
        self.n_params = n_params
        # self.x_new = torch.zeros((self.n_params.))

    # Policy net need to output acceleration. From there integration can compute state.
    # Projection onto value function computes new acceleration, then integrate to get state
    def forward(self, inputs):
        pos, vel = inputs[0:self.n_params.n_pos], inputs[self.n_params.n_pos:self.n_params.n_vel + self.n_params.n_pos]
        acc = self.policy(inputs)
        value = self.value(inputs)
        vel_new = self.integrate(vel, acc)
        pos_new = self.integrate(pos, vel)
        x_new = torch.vstack((pos_new, vel_new))

        # Compute network derivative w.r.t state
        d_value = torch.autograd.grad(
            [a for a in value], [inputs[0:inputs.size()[0]-self.n_params.n_desc]], create_graph=True, only_inputs=True
        )[0]

        # Return the policy(acc) projection onto value network
        return x_new - d_value * (F.relu((d_value*x_new).sum(dim=1))/(d_value**2).sum(dim=1))[:, None]

    def loss(self, input):



# Convert to dataset
d_train_d = TensorDataset(torch.Tensor(d_train.transpose()))
d_loader = DataLoader(d_train_d, batch_size=64, shuffle=True)
lr = 1e-4
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, policy_net.parameters()), lr=lr)
