import random
from recordtype import recordtype
import matplotlib.pyplot as plt
import numpy as np
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
device = 'cuda' if torch.cuda.is_available() else 'cpu'

GridParams = recordtype(
    'GridParams',
    'row_col def_cost min_cost cost_spread goal'
)


class GridWorld:
    def __init__(self, gp: GridParams):
        self.gp = gp
        self._grid = np.ones((gp.row_col[0], gp.row_col[1]))
        self._actions = {"up": [-1, 0], "down": [+1, 0], "left": [0, -1], "right": [0, +1]}

    def cost_grid(self):
        self._grid = self._grid * self.gp.def_cost
        for idx, cost in self.gp.cost_spread.items():
            self._grid[idx[0], idx[1]] = cost

        return self._grid

    def step(self, action: str, state: np.array):
        valid = lambda s: (0 <= s[0] < self.gp.row_col[0]) and \
                          (0 <= s[1] < self.gp.row_col[1])

        if np.linalg.norm(state-self.gp.goal) == 0.0:
            return state, 0

        cost = self._grid[state[0]][state[1]]
        act = self._actions[action]
        new_state = state + act
        if valid(new_state):
            state = new_state

        return state, cost


class ValueIteration:
    def __init__(self, world: GridWorld, actions: list, states: list, it_lim: int):
        self._world = world
        self._actions = actions
        self._states = states
        self._values = np.zeros_like(self._world.cost_grid())
        self._iteration_limit = it_lim

    def compute_values(self):
        for iteration in range(self._iteration_limit):
            for s1 in self._states[0]:
                for s2 in self._states[1]:
                    v = []
                    for a in self._actions:
                        state, cost = self._world.step(a, np.array([int(s1), int(s2)]))
                        n_v = cost + self._values[int(state[0]), int(state[1])]
                        v.append(n_v)

                    self._values[int(s1), int(s2)] = min(v)

        return self._values


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
        return z, self.decoder(z)


def train(autoencoder, data, epochs=100):
    opt = torch.optim.Adam(autoencoder.parameters(), 1e-5)
    for epoch in range(epochs):
        for cols in range(data.shape[1]):
            x = torch.tensor(data[:, cols], dtype=torch.float)
            x = x.to(device)
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl + ((x - x_hat)**2).sum()
            if epoch % 25 == 0:
                print(f"epoch: {epoch} loss: {loss} \n")
            loss.backward()
            opt.step()
    return autoencoder


def generate_values(autoencoder, r0, r1):
    values_data = []
    for x in r0:
        for y in r1:
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(gp.row_col[0], gp.row_col[1]).to('cpu').detach().numpy()
            values_data.append(([x, y], x_hat))
    return values_data


def plot_value(values: np.array, min_max: list, title: str):
    plt.figure()
    plt.imshow(
        values,
        interpolation=None,
        extent=[min_max[0], min_max[1]+1, min_max[1]+1, min_max[0]]
    )
    plt.title(title)
    plt.show()


if __name__ == "__main__":

    # Define world constant
    row, col = 10, 10
    total_size = row * col
    x = np.linspace(0, row - 1, row)
    y = np.linspace(0, col - 1, col)
    states = [x.tolist(), y.tolist()]
    actions = ["up", "down", "left", "right"]

    ex_per_sample = 1
    samples = int(row*col * 0.75)
    results = np.zeros((total_size, samples * ex_per_sample))
    perts = np.random.randint(1, 10, ex_per_sample)
    gp = GridParams(
            row_col=[row, col],
            def_cost=1,
            min_cost=0,
            cost_spread={(0, 0): 0},
            goal=np.array([0, 0])
        )

    gw = GridWorld(gp)

    for i in range(samples):
        # Sample IID goals
        min_bound, max_bound = 0, col - 1
        goal = np.random.randint(min_bound, max_bound, 2)
        for j in range(ex_per_sample):
            # Update gw
            gp.def_cost = 1
            gp.cost_spread = {(goal[0], goal[1]): 0}
            gp.goal = goal
            gw = GridWorld(gp)
            # Compute value map
            vi = ValueIteration(gw, actions, states, 40)
            values = vi.compute_values()
            results[:, i*ex_per_sample + j] = values.reshape(np.size(values))

    print("----------------Training VAE-----------------")
    vae = VariationalAutoencoder(2, total_size, [16]).to(device)
    vae = train(vae, results)
    value_data = generate_values(vae, x, y)

    while True:
        goal = input("What goal position")
        try:
            goal = goal.split(sep=",")
            goal = [int(goal[0]), int(goal[1])]
        except Exception as e:
            print("Wrong Goal")

        z = torch.Tensor([[goal[0], goal[1]]]).to(device)
        x_hat = vae.decoder(z)
        x_hat = x_hat.reshape(row, col).to('cpu').detach().numpy()

        plt.figure()
        plt.imshow(
            x_hat,
            interpolation=None,
            extent=[min(states[0]), max(states[0])+1, max(states[1])+1, min(states[1])]
        )

        plt.title(f"goal at {goal[0], goal[1]}")
        plt.show()
