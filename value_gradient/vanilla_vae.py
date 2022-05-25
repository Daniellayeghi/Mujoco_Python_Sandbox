import pandas as pd
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Decoder(nn.Module):
    def __init__(self, latent_dims, input_dim, hds):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, hds[0])
        self.linear2 = nn.Linear(hds[0], input_dim)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, input_dim, hds:list):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hds[0])
        self.linear2 = nn.Linear(hds[0], latent_dims)
        self.linear3 = nn.Linear(hds[0], latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, input_dims, hds):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, input_dims, hds)
        self.decoder = Decoder(latent_dims, input_dims, hds)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train(autoencoder, data, epochs=500):
    opt = torch.optim.SGD(autoencoder.parameters(), 1e-5)
    for epoch in range(epochs):
        for cols in range(data.shape[1]):
            x = torch.tensor(data[:, cols], dtype=torch.float)
            x = x.to(device)
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            if epoch % 25 == 0:
                print(f"epoch: {epoch} loss: {loss} \n")
            loss.backward()
            opt.step()
    return autoencoder


if __name__ == "__main__":
    parent = "/home/daniel/Repos/OptimisationBasedControl/data"
    f_sv = pd.read_csv(f"{parent}/di_data_size_313_data_num_19.csv").to_numpy()
    n_state, n_data, total_size = 2, int(f_sv.shape[1]/(2+1)), int(f_sv.shape[0] * (2+1))
    f_sv_c = np.zeros((total_size, n_data))

    for i in range(n_data):
        j = i * (n_state + 1)
        f_sv_c[:, i] = f_sv[:, j:j+n_state+1].reshape(total_size)


