import matplotlib.pyplot as plt
import numpy as np
import torch


if __name__ == "__main__":
    model = torch.nn.Sequential(
            torch.nn.Linear(2, 96),
            torch.nn.ReLU(),
            torch.nn.Linear(96, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Dropout(0.25)
        )

    model.load_state_dict(torch.load("state_dict_model.pt"))
    model.eval()

    disc = 100
    pos_arr = torch.linspace(-4*np.pi, np.pi*4, disc)
    vel_arr = torch.linspace(-4*np.pi, np.pi*4, disc)

    pos_value = np.zeros((disc, 1))
    vel_value = np.zeros((disc, 1))
    prediction = np.zeros((disc, disc))

    for pos in range(vel_arr.numpy().shape[0]):
        for vel in range(vel_arr.numpy().shape[0]):
            prediction[pos][vel] = model(
                torch.from_numpy(np.array([pos_arr[pos], vel_arr[vel]])).float()
            ).detach().numpy()[0]

    [P, V] = np.meshgrid(pos_arr.numpy(), vel_arr.numpy())

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(P, V, prediction, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('surface')
    ax.set_xlabel('Pos')
    ax.set_ylabel('Vel')
    plt.show()
    ax.set_zlabel('Value')

    # fig = plt.imshow(
    #     prediction,
    #     extent=[min(pos_arr), max(pos_arr), min(vel_arr), max(vel_arr)],
    #     origin="lower", interpolation='bilinear'
    # )

    plt.show()
