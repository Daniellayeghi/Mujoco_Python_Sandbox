import matplotlib.pyplot as plt
import numpy as np
import torch
from utilities.data_utils import *
from mlp_torch import MLP
from networks import ValueFunction
from net_utils_torch import LayerInfo
from value_gradient.utilities.torch_device import device
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Goal State')
    parser.add_argument('-l', '--goal', nargs='+', help='[pos, vel]', required=True, type=float)
    args = parser.parse_args()
    goal = args.goal

    # Data params
    batch_size = 16
    d_params = DataParams(3, 2, 1, 1, 1, 2, [1, 2], batch_size)

    # Networks and optimizers
    val_input, value_output = d_params.n_state, 1
    layer_dims = [val_input, 32, 64, 32, value_output]
    v_layers = [layer_dims, [], 0, [torch.nn.Softplus(), torch.nn.Softplus(), torch.nn.Softplus(), None]]
    value_net = ValueFunction(d_params, LayerInfo(*v_layers), False, 1).to(device)

    value_net.load_state_dict(torch.load("op_value_relu6.pt"))
    value_net.eval()

    # State encoder network
    x_encoder_input, x_encoder_output = d_params.n_state + d_params.n_desc, d_params.n_state
    x_encoder_layer_dims = [x_encoder_input, x_encoder_output]
    x_encoder_layers = [x_encoder_layer_dims, [], 0, [None, None]]
    x_encoder_net = MLP(LayerInfo(*x_encoder_layers), False, 1).to(device)

    x_encoder_net.load_state_dict(torch.load("./op_enc_relu6.pt"))

    for param in x_encoder_net.parameters():
        print(param)

    x_encoder_net.eval()
    disc = 100
    pos_arr = torch.linspace(-1, 1, disc)
    vel_arr = torch.linspace(-.3, .3, disc)

    pos_value = np.zeros((disc, 1))
    vel_value = np.zeros((disc, 1))
    value_matrix = np.zeros((disc, disc))

    for pos in range(pos_arr.numpy().shape[0]):
        for vel in range(vel_arr.numpy().shape[0]):
            value_matrix[pos][vel] = value_net(x_encoder_net(
                    torch.tensor([pos_arr[pos], vel_arr[vel], goal[0], goal[1]]).float().to(device)
            )).detach().cpu().numpy()[0]

            # value_matrix[pos][vel] = value_net(
            #         torch.tensor([pos_arr[pos] - goal[0], vel_arr[vel] - goal[1]]).float().to(device)
            # ).detach().cpu().numpy()[0]

    [P, V] = np.meshgrid(pos_arr.numpy(), vel_arr.numpy())

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(P, V, value_matrix, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('surface')
    ax.set_xlabel('Pos')
    ax.set_ylabel('Vel')
    plt.show()
    ax.set_zlabel('Value')

    plt.show()