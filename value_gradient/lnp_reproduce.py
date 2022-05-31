import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import TensorDataset, Dataset, DataLoader
from CVAE_torch import LayerInfo, Decoder, VariationalEncoder, CVAE, CVAEloss
import numpy as np
import pandas as pd

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


    def adjust_lr(optimizer, decay_rate=0.95):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_rate

    # Train network
    for epoch in range(1000):
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

