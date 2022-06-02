import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import TensorDataset, DataLoader
from CVAE_torch import LayerInfo, Decoder, VariationalEncoder, CVAE, CVAEloss
import numpy as np
import pandas as pd

if __name__ == "__main__":
    n_traj, n_states, n_train, n_cond = 5000, 2, int(5000*0.75), 4

    # Load path
    parent_path = "../../sr3-matlab/demo/"
    values = pd.read_csv(parent_path + "values_norm_di.csv", sep=',', header=None).to_numpy()
    states = pd.read_csv(parent_path + "traj_di.csv", sep=',', header=None).to_numpy()
    conds = pd.read_csv(parent_path + "conds_di.csv", sep=',', header=None).to_numpy() # init; goal
    input_q = np.vstack((states, values, conds))
    d_train = input_q[:, 0:n_train]
    d_test = input_q[:, n_train:]

    # Convert to dataset
    d_train_d = TensorDataset(torch.Tensor(d_train.transpose()))
    d_loader = DataLoader(d_train_d, batch_size=32, shuffle=True)

    # Create model
    e_layers, d_layers = [[757, 64, 6], [2], 0.5], [[7, 64, 753], [2], 0.5]
    decoder = Decoder(LayerInfo(*d_layers))
    encoder = VariationalEncoder(LayerInfo(*e_layers))
    cvae = CVAE(encoder, decoder)
    cvae = cvae.to(device)
    lr = 1e-4
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cvae.parameters()), lr=lr)

    # Train network
    for epoch in range(50000):
        train_loss, n = 0.0, 0
        for b_inq in d_loader:
            b_inq = b_inq[0].to(device)
            c_inp = b_inq[:, -n_cond:]
            mean, log_var, x_hat = cvae(b_inq, c_inp)
            loss = CVAEloss(mean, log_var, b_inq[:, 0:-n_cond], x_hat).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
            n += b_inq[:, 0:-n_cond].shape[0]

        train_loss /= n
        print(
            'epoch %d, train loss %.4f'
              % (epoch, train_loss)
        )

        if train_loss < 0.00035:
            break

    torch.save(cvae.state_dict(), "./di_value.pt")

    # Sample latent space
    cvae_trained = CVAE(VariationalEncoder(LayerInfo(*e_layers)), Decoder(LayerInfo(*d_layers)))
    cvae_trained.load_state_dict(torch.load("./di_value.pt"))
    cvae.eval()

    z = torch.Tensor(np.random.randn(1, int(e_layers[0][-1] / 2)))
    c = torch.Tensor(np.array([[0.7, 0, -0.7, 0]]))
    zc = torch.cat((z, c), dim=1)
    x_out = cvae_trained.sample(zc)
    k = False