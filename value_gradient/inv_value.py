from mlp_torch import MLP
from net_utils_torch import LayerInfo
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import TensorDataset, DataLoader
# import mujoco
# from mujoco import derivative


class OptimalPolicy(torch.nn.Module):
    def __init__(self, policy: MLP, value: MLP, integrate=lambda x1, x2, delta=.01: x1 + x2 * delta):
        super(OptimalPolicy, self).__init__()
        self.policy = policy
        self.value = value
        self.derivative = integrate

    # Policy net need to output acceleration. From there integration can compute state.
    # Projection onto value function computes new acceleration, then integrate to get state
    def forward(self, input):
        acc = self.policy(input)
        value = self.value(input)
        x_d = self.integrate(input, acc)
        d_value = torch.autograd.grad([a for a in value], [input], create_graph=True, only_inputs=True)[0]
        rv = x_d - d_value * (F.relu((d_value*x_d).sum(dim=1))/(d_value**2).sum(dim=1))[:,None]


n_pos, n_vel, n_desc = 1, 1, 1
n_state = n_pos + n_vel

# Value network
val_input, value_output = n_state + n_desc, 1
v_layers = [[val_input, 16, value_output], [], 0]
value_net = MLP(LayerInfo(*v_layers))

# Policy network
policy_input, policy_output = n_state + n_desc, n_pos
p_layers = [[val_input, 16, 16, value_output], [], 0]
policy_net = MLP(LayerInfo(*p_layers))

# Load Data
n_traj, n_train, n_cond = 5000, int(5000 * 0.75), 4
parent_path = "../../sr3-matlab/demo/"
values = pd.read_csv(parent_path + "di_values.csv", sep=',', header=None).to_numpy()
states = pd.read_csv(parent_path + "di_states.csv", sep=',', header=None).to_numpy()
conds = pd.read_csv(parent_path + "di_conds.csv", sep=',', header=None).to_numpy()  # init; goal
input_q = np.vstack((states, values, conds))
d_train = input_q[:, 0:n_train]
d_test = input_q[:, n_train:]

# Convert to dataset
d_train_d = TensorDataset(torch.Tensor(d_train.transpose()))
d_loader = DataLoader(d_train_d, batch_size=128, shuffle=True)
lr = 1e-4
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cvae.parameters()), lr=lr)

