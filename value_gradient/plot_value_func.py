import matplotlib.pyplot as plt
import numpy as np
import torch
from utilities.data_utils import *
from networks import ValueFunction
from net_utils_torch import LayerInfo
from torch_device import device, is_cuda


if __name__ == "__main__":
    batch_size = 64
    d_params = DataParams(3, 2, 1, 1, 1, 0, [1, 2], batch_size)

    # Networks and optimizers
    val_input, value_output = d_params.n_state, 1
    layer_dims = [val_input, 16, 16, value_output]
    v_layers = [layer_dims, [], 0, [torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU(), None]]
    model = ValueFunction(d_params, LayerInfo(*v_layers)).to(device)

    model.load_state_dict(torch.load("op_value_di6.pt"))
    model.eval()

    disc = 100
    pos_arr = torch.linspace(-3, 3, disc)
    vel_arr = torch.linspace(-1, 1, disc)

    pos_value = np.zeros((disc, 1))
    vel_value = np.zeros((disc, 1))
    prediction = np.zeros((disc, disc))

    for pos in range(pos_arr.numpy().shape[0]):
        for vel in range(vel_arr.numpy().shape[0]):
            prediction[pos][vel] = model(
                torch.from_numpy(np.array([pos_arr[pos], vel_arr[vel]])).float().to(device)
            ).detach().cpu().numpy()[0]

    [P, V] = np.meshgrid(pos_arr.numpy(), vel_arr.numpy())

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(P, V, prediction, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('surface')
    ax.set_xlabel('Pos')
    ax.set_ylabel('Vel')
    plt.show()
    ax.set_zlabel('Value')

    plt.show()
