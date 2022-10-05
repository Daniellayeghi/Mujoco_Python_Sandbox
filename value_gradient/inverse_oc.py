import sys
import numpy as np
from mlp_torch import MLP
from net_utils_torch import LayerInfo
from utilities.data_utils import *
from ioc_loss_di import *
from networks import ValueFunction
from torch_device import device, is_cuda
import mujoco
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import shuffle
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-t', '--train', help='Train NN?', required=True, type=str, default='n')
    parser.add_argument('-bs', '--batch_size', help="Batch size", required=True, type=int, default=16)
    parser.add_argument('-p', '--path', help="Path to Data", required=True, type=str, default='\n')
    args = parser.parse_args()
    TRAIN = args.train == 'y'
    batch_size = args.batch_size
    data_path = args.path

    # Mujoco models
    d_params = DataParams(3, 2, 1, 1, 1, 2, [1, 2], batch_size)
    data = pd.read_csv(data_path, sep=',', header=None).to_numpy()
    data = shuffle(data)[0:int(data.shape[0] * .7), :]
    # ind = np.argsort(data[:, 2])
    # data = data[ind, :]

    n_train = int(data.shape[0] * 0.7)
    d_train = torch.Tensor(data[0:n_train, :]).to(device)
    d_test = torch.Tensor(data[n_train:, :]).to(device)

    d_train_d = TensorDataset(d_train)
    d_test = TensorDataset(d_test)
    d_loader = DataLoader(d_train_d, batch_size=d_params.n_batch, shuffle=True, drop_last=True)
    d_loader_test = DataLoader(d_test, batch_size=d_params.n_batch, shuffle=True, drop_last=True)

    # Networks and optimizers
    val_input, value_output = d_params.n_state, 1
    layer_dims = [val_input,32, 64, 32, value_output]
    v_layers = [layer_dims, [], 0, [torch.nn.Softplus(), torch.nn.Softplus(), torch.nn.Softplus(), None]]
    value_net = MLP(LayerInfo(*v_layers), False, 1).to(device)

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

    class OptPolicy(torch.nn.Module):
        def __init__(self, feature_net: MLP, value_net: MLP, RinvB: torch.Tensor, params: DataParams):
            super(OptPolicy, self).__init__()
            self.feature_net = feature_net
            self.value_net = value_net
            self.RinvB = RinvB
            self.params = params

        def forward(self, x_external):
            features = self.feature_net(x_external).requires_grad_()
            value = self.value_net(features).requires_grad_()

            dvdx = torch.autograd.grad(
                value, features, grad_outputs=torch.ones_like(value), create_graph=True, only_inputs=True
            )[0].requires_grad_().view(self.params.n_batch, 1, self.params.n_state)

            u = -torch.bmm(b_B_R, dvdx.mT)

            return u

    b_Rinv = torch.ones((batch_size, 1, 1)) * 1/50
    b_Btran = torch.tensor([0.0005, 0.09999975]).repeat(d_params.n_batch, 1, 1)
    b_B_R = 0.5 * torch.bmm(b_Rinv, b_Btran).to(device)
    goal = torch.zeros((1, d_params.n_state)).to(device)

    pi = OptPolicy(x_encoder_net, value_net, b_B_R, d_params)

    lr = 3e-4
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, pi.parameters()), lr=lr)

    mse_loss = torch.nn.MSELoss()

    def b_l2_loss(x_external, u_star):
        l1_enc = sum(p.abs().sum() for p in pi.feature_net.parameters()) * 0.0001 * 1
        l_ioc = u_star.view(d_params.n_batch, d_params.n_ctrl, 1) - pi(x_external)
        loss = torch.mean(l_ioc.square().sum(2))
        return loss + l1_enc

    def save_models(value_path: str, encoder_path: str):
        print("########## Saving Trace ##########")
        torch.save(value_net.state_dict(), value_path + ".pt")
        torch.save(x_encoder_net.state_dict(), encoder_path + ".pt")

    value_net.train()
    x_encoder_net.train()

    if TRAIN:
        try:
            iter = 100
            for epoch in range(300):
                running_loss = 0
                now = time.time()
                for i, d_t in enumerate(d_loader):
                    # TODO: Maybe this copy is unnecessary
                    x_external = (d_t[0][:, :-d_params.n_ctrl]).requires_grad_()
                    u_star = d_t[0][:, d_params.n_state + d_params.n_desc:]
                    loss = b_l2_loss(x_external, u_star)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    running_loss += loss.detach() * len(d_t)
                    if not (i % iter):
                        print(f"Batch completion train: {int(i / len(d_loader) * 100)}%,", end="\r")
                end = time.time()

                valid_loss = 0.0
                for i, d_v in enumerate(d_loader_test):
                    x_external = (d_v[0][:, :-d_params.n_ctrl]).requires_grad_()
                    u_star = d_v[0][:, d_params.n_state + d_params.n_desc:]
                    loss = b_l2_loss(x_external, u_star)
                    valid_loss += loss.detach() * len(d_t)
                    if not (i % iter):
                        print(f"Batch completion valid: {int(i / len(d_loader_test) * 100)}%,", end="\r")

                epoch_loss_train = running_loss / len(d_loader)
                epoch_loss_test = valid_loss / len(d_test)

                print('train loss: {} epoch: {} lr: {}'.format(epoch_loss_train.item(), epoch, optimizer.param_groups[0]['lr']))
                print('valid loss: {} epoch: {} lr: {}'.format(epoch_loss_test.item(), epoch, optimizer.param_groups[0]['lr']))
                print(f'bathc time: {end-now}')

                for param in x_encoder_net.parameters():
                    print(param)

            save_models("./op_value_relu6", "./op_enc_relu6")

        except KeyboardInterrupt:
            save_models("./op_value_relu6", "./op_enc_relu6")

    else:
        error = []
        value_net.load_state_dict(torch.load("./op_value_relu6.pt"))
        value_net.eval()
        for d in d_loader:
            x = d[0][:, :-1].requires_grad_()
            u = d[0][:, -1].requires_grad_()
            x = torch.vstack((x[:, 0] - x[:, 2], x[:, 1] - x[:, 3])).T.requires_grad_()
            bloss = u.view(d_params.n_batch, d_params.n_ctrl, 1) - ctrl(x, value_net)
            error.append(torch.mean(bloss.square().sum(2)).cpu().detach().numpy())

        print(np.mean(error))
