import numpy as np
import pandas as pd
import torch
from value_gradient.utilities.torch_device import device
from mlp_torch import MLP
from net_utils_torch import LayerInfo
from utilities.data_utils import *
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader

# Load Data
parent_path = "~/Desktop/data/"
d_params = DataParams(3, 2, 1, 1, 1, 2, [1, 2], 64)

data = pd.read_csv(parent_path + "di_matlab_data.csv", sep=',', header=None).to_numpy()
data = shuffle(np.vstack(data))[0:int(data.shape[0] * .01), :]

x = torch.Tensor(data[:, :-1]).to(device)
x = TensorDataset(x)
x_loader = DataLoader(x, batch_size=64, shuffle=True, drop_last=True)

# State encoder network
x_encoder_input, x_encoder_output = d_params.n_state + d_params.n_desc, d_params.n_state
x_encoder_layer_dims = [x_encoder_input, x_encoder_output]
x_encoder_layers = [x_encoder_layer_dims, [], 0, [None, None]]
x_encoder_net = MLP(LayerInfo(*x_encoder_layers), False, 1).to(device)

# state decoder network
x_decoder_in, x_decoder_out = d_params.n_state, d_params.n_state + d_params.n_desc
x_decoder_layer_dims = [x_decoder_in, x_decoder_out]
x_decoder_layers = [x_decoder_layer_dims, [], 0, [None, None]]
x_decoder_net = MLP(LayerInfo(*x_decoder_layers), False, 1).to(device)

fmse = torch.nn.MSELoss()


def floss(x_external):
    pos_error = x_external[:, 0] - x_external[:, 2]
    vel_error = x_external[:, 1] - x_external[:, 3]
    target = torch.vstack((pos_error, vel_error)).T
    x_enc = x_encoder_net(x_external).requires_grad_()
    x_dec = x_decoder_net(x_enc).requires_grad_()
    lmse_auto = fmse(x_external, x_dec) * 0
    lmse_recon = fmse(target, x_enc)
    l1_lambda = 0.0001
    l1_enc = sum(p.abs().sum() for p in x_encoder_net.parameters()) * l1_lambda * 1
    l1_dec = sum(p.abs().sum() for p in x_decoder_net.parameters()) * l1_lambda * 0
    loss = lmse_recon + lmse_auto + l1_enc + l1_dec
    return loss


lr = 1e-4
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, x_encoder_net.parameters()), lr=lr)


try:
    iter = 500
    for epoch in range(300):
        running_loss = 0
        for i, d_t in enumerate(x_loader):
            # TODO: Maybe this copy is unnecessary
            x_external = d_t[0].requires_grad_()
            l = floss(x_external)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += l.detach() * len(d_t)
            if not (i % iter):
                print(f"Batch completion train: {int(i / len(x_loader) * 100)}%,", end="\r")

        epoch_loss_train = running_loss / len(x_loader)
        print('train loss: {} epoch: {} lr: {}'.format(epoch_loss_train.item(), epoch, optimizer.param_groups[0]['lr']))

    for p in x_encoder_net.parameters():
        print(p)
        print(p.abs().sum())
except KeyboardInterrupt:
    for p in x_encoder_net.parameters():
        print(p)
        print(p.abs().sum())

    print("########## Saving Trace ##########")
    torch.save(x_encoder_net.state_dict(), "./diff" + ".pt")
