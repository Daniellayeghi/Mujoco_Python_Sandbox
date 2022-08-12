import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.utils
import torch.distributions
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from net_utils_torch import LayerInfo, InitNetwork


class Decoder(nn.Module):
    def __init__(self, layer_info: LayerInfo, apply_sigmoid=False):
        super(Decoder, self).__init__()
        self.decoder = InitNetwork(layer_info).get_network()
        self.apply_sigmoid = apply_sigmoid

    def forward(self, zc_inp):
        logits = self.decoder(zc_inp)
        if self.apply_sigmoid:
            probs = torch.sigmoid(logits)
            return probs
        return logits


class VariationalEncoder(nn.Module):
    def __init__(self, layer_info: LayerInfo):
        super(VariationalEncoder, self).__init__()
        self.encoder = InitNetwork(layer_info).get_network()

    def reparameterize(self, mean, log_var):
        eps = torch.normal(torch.zeros_like(mean))
        return eps * torch.exp(log_var * 0.5) + mean

    def forward(self, xc_inq):
        z = self.encoder(xc_inq)
        mean, log_var = torch.Tensor.split(z, 3, dim=1)
        return mean, log_var, self.reparameterize(mean, log_var)


class CVAE(nn.Module):
    def __init__(self, encoder: VariationalEncoder, decoder: Decoder):
        super(CVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def sample(self, zc_inp, apply_sigmoid=False):
        if apply_sigmoid:
            return torch.sigmoid(self.decoder(zc_inp))
        return self.decoder(zc_inp)

    def forward(self, xc_inq, c_inp):
        mean, log_var, z = self.encoder(xc_inq)
        zc_inp = torch.cat((z, c_inp), dim=1)
        return mean, log_var, self.decoder(zc_inp)


def CVAEloss(mean, logvar, x, x_hat):
    mse = nn.MSELoss()
    recon = mse(x_hat, x)
    kl_loss = 10 ** -4 * 2 * torch.sum(torch.exp(logvar) + mean ** 2 - 1. - logvar, 1)
    return torch.sum(recon + kl_loss)