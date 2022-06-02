import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import TensorDataset, DataLoader
from CVAE_torch import LayerInfo, Decoder, VariationalEncoder, CVAE, CVAEloss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


if __name__ == "__main__":
    epochs, it, numTrain, mb_size, dim, dimW, gridSize = 201, 0, 7000, 256, 6, 3, 11
    dataElements = dim + 3 * 3 + 2 * dim

    # Load data
    parent_path = "../../LearnedSamplingDistributions/"
    occ_grid = np.load(parent_path + "occ_grid.npy")
    XC = pd.read_csv(parent_path + "narrowDataFile.txt", sep=',', header=None).to_numpy()
    XC = np.delete(XC, -1, axis=1)

    numEntries = XC.shape[0]
    X_train = XC[0:numTrain, 0:dim]
    X_test = XC[numTrain:numEntries, 0:dim]
    C_test = XC[numTrain:numEntries, dim:dataElements]
    cs = np.concatenate(
        (XC[0:numEntries, dim + 3 * dimW:dataElements], occ_grid), axis=1
    )

    c_dim = cs.shape[1]
    c_gapsInitGoal = C_test
    C_train = cs[0:numTrain, :]
    C_test = cs[numTrain:numEntries, :]

    # Convert to dataset
    XC_train = np.concatenate((X_train, C_train), axis=1)
    XC_train_d = TensorDataset(torch.Tensor(XC_train))
    XC_loader = DataLoader(XC_train_d, batch_size=256, shuffle=True)

    # Build network
    e_layers, d_layers = [[139, 512, 512, 6], [2], 0.5], [[136, 512, 512, 6], [2], 0.5]
    decoder = Decoder(LayerInfo(*d_layers))
    encoder = VariationalEncoder(LayerInfo(*e_layers))
    cvae = CVAE(encoder, decoder)
    cvae = cvae.to(device)
    lr = 1e-4
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cvae.parameters()), lr=lr)

    # Train network
    for epoch in range(50000):
        train_loss, n = 0.0, 0
        for b_inq in XC_loader:
            b_inq = b_inq[0].to(device)
            c_inp = b_inq[:, dim:]
            mean, log_var, x_hat = cvae(b_inq, c_inp)
            loss = CVAEloss(mean, log_var, b_inq[:, 0:dim], x_hat).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
            n += b_inq[:, 0:dim].shape[0]

        train_loss /= n
        print(
            'epoch %d, train loss %.4f'
              % (epoch, train_loss)
        )
        if train_loss < 0.0045:
            break

    # Build grid
    occGridSamples = np.zeros([gridSize * gridSize, 2])
    gridPointsRange = np.linspace(0, 1, num=gridSize)

    idx = 0
    for i in gridPointsRange:
        for j in gridPointsRange:
            occGridSamples[idx, 0] = i
            occGridSamples[idx, 1] = j
            idx += 1

    num_viz, numTest = 3000, X_test.shape[0]
    vizIdx = np.load(parent_path + "vizIdx.npy")
    c_sample_seed = C_test[vizIdx, :]
    c_sample = np.repeat([c_sample_seed], num_viz, axis=0)
    c_viz = c_gapsInitGoal[vizIdx, :]
    # Sample latent space
    num_viz = 3000
    z = torch.Tensor(np.random.randn(num_viz, int(d_layers[0][-1]/2)))
    np.save(parent_path + "z.npy", z)
    z = z.to(device)
    c_sample_t = torch.Tensor(c_sample).to(device)
    zc_inp = torch.cat((z, c_sample_t), dim=1)
    y_viz = cvae.sample(zc_inp)

    fig1 = plt.figure(figsize=(10, 6), dpi=80)
    ax1 = fig1.add_subplot(111, aspect='equal')

    plt.scatter(y_viz[:, 0].cpu().detach().numpy(), y_viz[:, 1].cpu().detach().numpy(), color="green", s=70, alpha=0.1)

    dw, dimW = 0.1, 3
    gap1 = c_viz[0:3]
    gap2 = c_viz[3:6]
    gap3 = c_viz[6:9]
    init = c_viz[9:15]
    goal = c_viz[15:21]

    obs1 = [0, gap1[1] - dw, -0.5, gap1[0], gap1[1], 1.5]
    obs2 = [gap2[0] - dw, 0, -0.5, gap2[0], gap2[1], 1.5]
    obs3 = [gap2[0] - dw, gap2[1] + dw, -0.5, gap2[0], 1, 1.5]
    obs4 = [gap1[0] + dw, gap1[1] - dw, -0.5, gap3[0], gap1[1], 1.5]
    obs5 = [gap3[0] + dw, gap1[1] - dw, -0.5, 1, gap1[1], 1.5]
    obsBounds = [-0.1, -0.1, -0.5, 0, 1.1, 1.5,
                 -0.1, -0.1, -0.5, 1.1, 0, 1.5,
                 -0.1, 1, -0.5, 1.1, 1.1, 1.5,
                 1, -0.1, -0.5, 1.1, 1.1, 1.5, ]

    obs = np.concatenate((obs1, obs2, obs3, obs4, obs5, obsBounds), axis=0)
    for i in range(0, int(obs.shape[0] / (2 * dimW))):
        ax1.add_patch(
            patches.Rectangle(
                (obs[i * 2 * dimW], obs[i * 2 * dimW + 1]),  # (x,y)
                obs[i * 2 * dimW + dimW] - obs[i * 2 * dimW],  # width
                obs[i * 2 * dimW + dimW + 1] - obs[i * 2 * dimW + 1],  # height
                alpha=0.6
            ))

    for i in range(0, gridSize * gridSize):  # plot occupancy grid
        cIdx = i + 2 * dim
        if c_sample_seed[cIdx] == 0:
            plt.scatter(occGridSamples[i, 0], occGridSamples[i, 1], color="red", s=50, alpha=0.7)
        else:
            plt.scatter(occGridSamples[i, 0], occGridSamples[i, 1], color="green", s=50, alpha=0.7)

    plt.scatter(init[0], init[1], color="red", s=250, edgecolors='black')  # init
    plt.scatter(goal[0], goal[1], color="blue", s=250, edgecolors='black')  # goal

    plt.show()
