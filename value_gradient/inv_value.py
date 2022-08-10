from mlp_torch import MLP
from net_utils_torch import LayerInfo
from utilities import mj_utils
from utilities.torch_utils import to_variable
from collections import namedtuple
from mujoco import MjModel, MjData
from mujoco import derivative
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

d_train = to_variable(torch.Tensor(data[0:n_train, :]), torch.cuda.is_available())
d_test = data[n_train:, :]
# for value derivative
d_train.requires_grad = True
print(d_train.requires_grad)
d_train_d = TensorDataset(d_train)
d_loader = DataLoader(d_train_d, batch_size=2, shuffle=True)
DataParams = namedtuple('DataParams', 'n_state, n_pos, n_vel, n_ctrl, n_desc, n_batch')
d_params = DataParams(2, 1, 1, 1, 2, d_loader.batch_size)

# Value network
val_input, value_output = d_params.n_state + d_params.n_desc, d_params.n_ctrl
v_layers = [[val_input, 16, value_output], [], 0]
value_net = MLP(LayerInfo(*v_layers)).to(device)

# Policy network
policy_input, policy_output = d_params.n_state + d_params.n_desc, d_params.n_pos
p_layers = [[val_input, 16, 16, value_output], [], 0]
policy_net = MLP(LayerInfo(*p_layers)).to(device)

# Derivative values
m = MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml")
d = MjData(m)
d_vec = derivative.MjDataVecView(m, d)
params = derivative.MjDerivativeParams(1e-6, derivative.Wrt.Ctrl, derivative.Mode.Fwd)
du = derivative.MjDerivative(m, params)


class ValueFunction(MLP):
    def __init__(self, data_params: DataParams, layer_info: LayerInfo, apply_sigmoid=False):
        super(ValueFunction, self).__init__(layer_info, apply_sigmoid)
        self.loss = list()
        self._data_params = data_params
        self._lie_v_wrt_f = torch.zeros()
        self._lie_v_wrt_u = torch.zeros()
        self._value_derivative = torch.zeros()


class OptimalPolicy(torch.nn.Module):
    def __init__(
            self, policy: MLP, value: MLP, params: DataParams,
            derivative=lambda x1, x2, dt=.01: (x2 - x1) / dt,
            integrate=lambda x1, x2, dt=0.01: x1 + x2 * dt
    ):
        super(OptimalPolicy, self).__init__()
        self._policy_net = policy
        self._value_net = value
        self._params = params
        self._derivative = derivative
        self._integrate = integrate
        self._final_pos = torch.zeros((self._params.n_batch, self._params.n_pos)).to(device)
        self._final_state = torch.zeros((self._params.n_batch, self._params.n_vel * 2)).to(device)
        self._final_state_d = torch.zeros((self._params.n_batch, self._params.n_vel * 2)).to(device)
        self._value = to_variable(torch.zeros((self._params.n_batch, 1)))
        self._value_d = to_variable(torch.zeros((self._params.n_batch, self._params.n_state)))

    # Policy net need to output positions. From there derivatives can compute velocity.
    # Projection onto value function computes new acceleration, then integrate to get state
    def forward(self, inputs):
        pos, vel = inputs[:, 0:self._params.n_pos], inputs[:, self._params.n_pos:self._params.n_vel + self._params.n_pos]
        self._final_pos = self._policy_net(inputs)
        self._value = self._value_net(inputs)
        self._final_state_d[:, :self._params.n_vel] = self._derivative(pos, self._final_pos)
        self._final_state_d[:, self._params.n_vel:] = self._derivative(
            vel, self._final_state_d[:, :self._params.n_vel]
        )

        # Compute network derivative w.r.t state
        self._value_d = torch.autograd.grad(
            [a for a in self._value], [inputs], create_graph=True, only_inputs=True
        )[0][:, :self._params.n_state]

        # Return the new state form
        self._final_state_d = self._final_state_d - self._value_d * (F.relu(
            (self._value_d * self._final_state_d).sum(dim=1)
        )/(self._value_d ** 2).sum(dim=1))[:, None]

        self._final_state[:, self._params.n_vel:] = self._final_state_d[:, self._params.n_vel:]
        self._final_state[:, :self._params.n_vel] = self._integrate(pos, self._final_vel)

        return self._final_state, self._final_state_d


opt_policy = OptimalPolicy(policy_net, value_net, d_params).to(device)
lr = 1e-4
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, policy_net.parameters()), lr=lr)
for d in d_loader:
    # TODO: Maybe this copy is unnecessary
    x_c = to_variable(d[0][:, :-d_params.n_ctrl], torch.cuda.is_available())
    opt_policy(x_c)
