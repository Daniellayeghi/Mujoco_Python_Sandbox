from net_utils_torch import LayerInfo
from utilities.data_utils import *
from networks import ValueFunction
import torch
import numpy as np
import matplotlib.pyplot as plt


batch_size = 20
d_params = DataParams(3, 2, 1, 1, 1, 2, [1, 2], batch_size)
val_input, value_output = d_params.n_state + d_params.n_desc, d_params.n_ctrl
layer_dims = [val_input, 16, value_output]
v_layers = [layer_dims, [], 0, [torch.nn.ReLU() for _ in range(len(layer_dims) - 1)]]
value_net = ValueFunction(d_params, LayerInfo(*v_layers))
value_net.load_state_dict(torch.load("./op_value.pt"))
value_net.eval()

goal_vel = 0
goal_pos = 0
pos = torch.linspace(-1, 1, 10)
vel = torch.linspace(-1, 1, 10)
values = np.zeros(10)
fig = plt.figure()
ax = plt.axes(projection='3d')

for coords in range(pos.size()[0]):
    x_pos = pos[coords]
    x_vel = vel[coords]
    x_goal = torch.tensor([x_pos, x_vel, goal_pos, goal_vel])
    value = value_net(x_goal)
    values[coords] = value.detach().numpy()[0]

ax.plot3D(pos, vel, values)
plt.show()

    