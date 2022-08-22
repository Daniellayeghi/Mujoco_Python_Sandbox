from mlp_torch import MLP
from net_utils_torch import LayerInfo
from utilities.torch_utils import to_variable
from utilities.data_utils import *
from networks import ValueFunction, OptimalPolicy
import pandas as pd
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import TensorDataset, DataLoader


d_params = DataParams(3, 2, 1, 1, 1, 2, [], 2)


# Value network
val_input, value_output = d_params.n_state + d_params.n_desc, d_params.n_ctrl
v_layers = [[val_input, 16, value_output], [], 0]
value_net = ValueFunction(d_params, LayerInfo(*v_layers)).to(device)

policy_input, policy_output = d_params.n_state + d_params.n_desc, d_params.n_pos
p_layers = [[val_input, 16, 16, value_output], [], 0]
policy_net = MLP(LayerInfo(*p_layers)).to(device)

opt_policy = OptimalPolicy(policy_net, value_net, d_params).to(device)
lr = 1e-4
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, policy_net.parameters()), lr=lr)

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

for d in d_loader:
    # TODO: Maybe this copy is unnecessary
    x_c = to_variable(d[0][:, :-d_params.n_ctrl], torch.cuda.is_available())
    opt_policy(x_c)
