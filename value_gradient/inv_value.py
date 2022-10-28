import sys
from mlp_torch import MLP
from net_utils_torch import LayerInfo
from utilities.data_utils import *
from net_loss_functions import *
from task_loss_functions import *
from networks import ValueFunction, OptimalPolicy
from value_gradient.utilities.torch_device import device
import mujoco
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# Data params
batch_size = 20
d_params = DataParams(3, 2, 1, 1, 1, 2, [1, 2], batch_size)

# Mujoco models
m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml")
d = mujoco.MjData(m)
batch_op = MjBatchOps(m, d_params)

# Value network
# TODO output layer must be RELU for > +
val_input, value_output = d_params.n_state + d_params.n_desc, d_params.n_ctrl
layer_dims = [val_input, 16, value_output]
v_layers = [layer_dims, [], 0, [torch.nn.ReLU() for _ in range(len(layer_dims) - 1)]]
value_net = ValueFunction(d_params, LayerInfo(*v_layers)).to(device)

policy_input, policy_output = d_params.n_state + d_params.n_desc, d_params.n_pos
p_layers = [[val_input, 16, 16, value_output], [], 0, [torch.nn.ReLU(), torch.nn.ReLU(), None]]
policy_net = MLP(LayerInfo(*p_layers)).to(device)

opt_policy = OptimalPolicy(policy_net, value_net, d_params).to(device)
lr = 1e-4
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, policy_net.parameters()), lr=lr)

# Load Data
parent_path = "../../OptimisationBasedControl/data/"
data = pd.read_csv(parent_path + "di_data_value.csv", sep=',', header=None).to_numpy()
n_traj, n_train = 5000, int(5000 * 0.75)
d_train = torch.Tensor(data[0:n_train, :]).to(device)
d_test = data[n_train:, :]

# for value derivative
d_train_d = TensorDataset(d_train)
d_loader = DataLoader(d_train_d, batch_size=d_params.n_batch, shuffle=True, drop_last=True)

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


def b_full_loss(x_desc_next: torch.Tensor,
                x_desc_curr: torch.Tensor,
                x_full_next: torch.Tensor,
                u_star: torch.Tensor,
                goal: torch.Tensor):
    clone = torch.mean(clone_loss(x_full_next, u_star))
    effort = torch.mean(effort_loss(x_full_next))
    dvdfdu = torch.mean(value_lie_loss(x_desc_next, u_star))
    dvdt = torch.mean(value_dt_loss_auto(x_desc_next, x_desc_curr))
    goal = torch.mean(value_goal_loss(goal))

    return torch.mean(clone + effort + dvdfdu + dvdt + goal), (clone, effort, dvdfdu, dvdt, goal)


running_loss = 0
goals = torch.zeros((d_params.n_batch, d_params.n_state + d_params.n_desc), requires_grad=False).to(device)

try:
    for epoch in range(150):
        for i, d in enumerate(d_loader):
            # TODO: Maybe this copy is unnecessary
            x_desc_curr = (d[0][:, :-d_params.n_ctrl]).requires_grad_()
            goals[:, :d_params.n_pos] = d[0][:, d_params.n_state + 1:-d_params.n_ctrl]
            goals[:, d_params.n_state:] = d[0][:, d_params.n_state:-d_params.n_ctrl]
            u_star = d[0][:, d_params.n_state + d_params.n_desc:]
            x_full_next, x_desc_next = opt_policy(x_desc_curr)
            value_net.update_grads(x_desc_next)
            # Detach x_desc_curr so its derivatives are ignored
            loss, each_loss = b_full_loss(x_desc_next, x_desc_curr.detach(), x_full_next, u_star, goals)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 20 == 0:
                last_loss = running_loss / 20
                print('batch {} loss: {} epoch{}'.format(i + 1, last_loss, epoch))
                print(
                    f"clone {each_loss[0]}, effort {each_loss[1]},"
                    f" dvdfdu {each_loss[2]}, dvdt {each_loss[3]},"
                    f" goal {each_loss[4]}"
                )
                running_loss = 0.

    stored_exception = sys.exc_info()
    print("########## Saving Trace ##########")
    torch.save(opt_policy.state_dict(), "./op_policy.pt")
    torch.save(value_net.state_dict(), "./op_value.pt")

except KeyboardInterrupt:
    stored_exception = sys.exc_info()
    print("########## Saving Trace ##########")
    torch.save(opt_policy.state_dict(), "./op_policy.pt")
    torch.save(value_net.state_dict(), "./op_value.pt")
