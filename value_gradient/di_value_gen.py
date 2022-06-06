import sys
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import TensorDataset, DataLoader
from CVAE_torch import LayerInfo, Decoder, VariationalEncoder, CVAE, CVAEloss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRAIN = False


if __name__ == "__main__":
    n_traj, n_states, n_train, n_cond = 5000, 2, int(5000*0.75), 4

    # Load path
    parent_path = "../../sr3-matlab/demo/"
    values = pd.read_csv(parent_path + "di_values.csv", sep=',', header=None).to_numpy()
    states = pd.read_csv(parent_path + "di_states.csv", sep=',', header=None).to_numpy()
    conds = pd.read_csv(parent_path + "di_conds.csv", sep=',', header=None).to_numpy() # init; goal
    input_q = np.vstack((states, values, conds))
    d_train = input_q[:, 0:n_train]
    d_test = input_q[:, n_train:]

    # Convert to dataset
    d_train_d = TensorDataset(torch.Tensor(d_train.transpose()))
    d_loader = DataLoader(d_train_d, batch_size=128, shuffle=True)

    # Create model
    e_layers, d_layers = [[757, 256, 6], [2], 0.5], [[7, 256, 753], [2], 0.5]
    decoder = Decoder(LayerInfo(*d_layers))
    encoder = VariationalEncoder(LayerInfo(*e_layers))
    cvae = CVAE(encoder, decoder)
    cvae = cvae.to(device)
    lr = 1e-4
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cvae.parameters()), lr=lr)

    if TRAIN:
        try:
            # Train network
            for epoch in range(50000):
                if not TRAIN:
                    break
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
        except KeyboardInterrupt:
            stored_exception = sys.exc_info()
            print("########## Saving Trace ##########")
            torch.save(cvae.state_dict(), "./di_value.pt")

    # Sample latent space
    cvae_trained = CVAE(VariationalEncoder(LayerInfo(*e_layers)), Decoder(LayerInfo(*d_layers)))
    cvae_trained.load_state_dict(torch.load("./di_value.pt"))
    cvae.eval()

    n_samples, z_size = 10, int(e_layers[0][-1] / 2)
    z = torch.Tensor(np.random.randn(n_samples, z_size))
    c = torch.Tensor(np.repeat(np.array([[0.71, 0, -0.733, 0]]), n_samples, axis=0))
    zc = torch.cat((z, c), dim=1)
    x_out = cvae_trained.sample(zc)
    pd.DataFrame.to_csv(pd.DataFrame(
        (x_out.cpu().detach().numpy()).transpose()), parent_path + "value_gen.csv", header=False, index=False
    )

    out = (x_out.cpu().detach().numpy()).transpose()
    div = int(out.shape[0]/(n_states + 1))
    x, xd, v = out[0: div, :], out[div: div*2, :], out[div*2:, :]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(x.shape[1]):
        ax.plot3D(x[:, i], xd[:, i], v[:, i], c=(np.random.random(), np.random.random(), np.random.random()))

    plt.show()
