from mlp_torch import MLP
from net_utils_torch import LayerInfo
import mujoco
from utilities.data_utils import *
from networks import ValueFunction, OptimalPolicy
from net_loss_functions import *
from task_loss_functions import *
import pandas as pd
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import TensorDataset, DataLoader

# Data params
batch_size = 10
d_params = DataParams(3, 2, 1, 1, 1, 2, [1, 2], batch_size)

# Mujoco models
m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml")
d = mujoco.MjData(m)
batch_op = MjBatchOps(m, d_params)

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
d_train = gradify(torch.Tensor(data[0:n_train, :]), torch.cuda.is_available()).requires_grad_()
d_test = data[n_train:, :]

# for value derivative
d_train_d = TensorDataset(d_train)
d_loader = DataLoader(d_train_d, batch_size=d_params.n_batch, shuffle=True)

# Set up networks loss and params
set_value_net__(value_net)
set_batch_ops__(batch_op)
set_dt_(0.01)
clone_loss = ctrl_clone_loss.apply
effort_loss = ctrl_effort_loss.apply
value_lie_loss = value_lie_loss.apply
value_time_loss = value_dt_loss.apply

# Setup task loss
x_gain = torch.diag(torch.tensor([10, .1])).repeat(d_params.n_batch, 1, 1)
set_gains__(x_gain)


def b_full_loss(x_desc_curr: torch.Tensor,
                x_desc_next: torch.Tensor,
                x_full_next: torch.Tensor,
                u_star: torch.Tensor,
                goal: torch.Tensor):
    return torch.mean(
        clone_loss(x_full_next, u_star) +
        # effort_loss(x_full_next) +
        # value_lie_loss(x_desc_next, u_star) +
        # value_time_loss(x_desc_next, x_desc_curr) +
        value_goal_loss(goal)
    )


running_loss = 0

for epoch in range(100):
    for i, d in enumerate(d_loader):
        goal = torch.zeros((d_params.n_batch, d_params.n_state + d_params.n_desc)).to(device)
        # TODO: Maybe this copy is unnecessary
        x_desc_curr = gradify(d[0][:, :-d_params.n_ctrl], torch.cuda.is_available())
        goal[:, :d_params.n_pos] = d[0][:, d_params.n_state + 1:-d_params.n_ctrl]
        goal[:, d_params.n_state:] = d[0][:, d_params.n_state:-d_params.n_ctrl]
        u_star = to_cuda(d[0][:, d_params.n_state + d_params.n_desc:])
        x_full_next, x_desc_next = opt_policy(x_desc_curr)
        x_next = gradify(x_full_next[:, :d_params.n_state])
        value_net.update_grads(x_desc_next)
        # Detach x_desc_curr so its derivatives are ignored
        loss = b_full_loss(x_desc_curr.detach(), x_desc_next, x_full_next, u_star, goal)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 100 == 0:
            last_loss = running_loss / 100
            print('batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

