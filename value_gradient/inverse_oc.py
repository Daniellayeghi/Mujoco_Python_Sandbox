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
    args = parser.parse_args()
    TRAIN = args.train == 'y'
    batch_size = args.batch_size

    # Mujoco models
    d_params = DataParams(3, 2, 1, 1, 1, 2, [1, 2], batch_size)
    parent_path = "/srv/data/daniel/ssd0/"
    data = pd.read_csv(parent_path + "data_di.csv", sep=',', header=None).to_numpy()
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
    layer_dims = [val_input, 64, 128, 64, value_output]
    v_layers = [layer_dims, [], 0, [torch.nn.Softplus(), torch.nn.Softplus(), torch.nn.Softplus(), None]]
    value_net = ValueFunction(d_params, LayerInfo(*v_layers), False, 1).to(device)

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

    lr = 3e-4
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, value_net.parameters()), lr=lr)

    set_value_net__(value_net)
    loss_v = loss_value_proj.apply

    b_Rinv = torch.ones((batch_size, 1, 1)) * 1/50
    b_Btran = torch.tensor([0.0005, 0.09999975]).repeat(d_params.n_batch, 1, 1)
    b_B_R = 0.5 * torch.bmm(b_Rinv, b_Btran).to(device)
    goal = torch.zeros((1, d_params.n_state)).to(device)

    def ctrl(x, value):
        v = value(x).requires_grad_()
        dvdx = torch.autograd.grad(
            v, x, grad_outputs=torch.ones_like(v), create_graph=True, only_inputs=True
        )[0].requires_grad_().view(d_params.n_batch, 1, d_params.n_state)

        u = -torch.bmm(b_B_R, dvdx.mT)
        return u

    mse_loss = torch.nn.MSELoss()

    def b_l2_loss(x_external, u_star):
        # pos_error = x_external[:, 0] - x_external[:, 2]
        # vel_error = x_external[:, 1] - x_external[:, 3]
        # x_enc = torch.vstack((pos_error, vel_error)).T
        x_enc = x_encoder_net(x_external).requires_grad_()
        # x_dec = x_decoder_net(x_enc).requires_grad_()
        # l_mse = mse_loss(x_external, x_dec) * 0
        l1_lambda = 0.0005
        l1_enc = sum(p.abs().sum() for p in x_encoder_net.parameters()) * l1_lambda * 1
        # l1_dec = sum(p.abs().sum() for p in x_decoder_net.parameters()) * 0
        l_ioc = u_star.view(d_params.n_batch, d_params.n_ctrl, 1) - ctrl(x_enc, value_net)
        loss = torch.mean(l_ioc.square().sum(2)) + l1_enc
        return loss

    def save_models(value_path: str, encoder_path: str):
        print("########## Saving Trace ##########")
        torch.save(value_net.state_dict(), value_path + ".pt")
        torch.save(x_encoder_net.state_dict(), encoder_path + ".pt")


    if TRAIN:
        try:
            iter = 100
            for epoch in range(500):
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
