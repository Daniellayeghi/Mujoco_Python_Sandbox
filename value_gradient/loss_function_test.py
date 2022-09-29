import numpy as np
import pandas as pd
from sklearn.utils import shuffle


# Load Data
parent_path = "../../OptimisationBasedControl/data/"

# data1 = np.load("di_data.npy")
data = []
for i in [1, 2]:
    name = f"di{i}"
    data_x = pd.read_csv(parent_path + name + "_state.csv", sep=',', header=None)
    data_g = pd.read_csv(parent_path + name + "_goal.csv", sep=',', header=None)
    data_u = pd.read_csv(parent_path + name + "_ctrl.csv", sep=',', header=None)
    data.append(np.hstack((data_x, data_g, data_u)))

data = np.vstack(data)
data = shuffle(np.vstack(data))[0:int(data.shape[0] * 1), :]
cost = []
us_star = []

R = np.array([1.73, 1, 1, 1.73]).reshape((2, 2))
B = np.array([0.0005, 0.09999975])


def ctrl(x):
    pos_error = x[0] - x[2]
    vel_error = x[1] - x[3]
    err = np.vstack((pos_error, vel_error))
    dJdx = 2 * err.T @ R
    u = np.clip(-0.5 * 1 * 1/50 * B @ dJdx.T, -1, 1)
    return u


for d in data:
    policy = d[-1]
    u_star = ctrl(d[:-1])
    cst = np.power((policy - u_star), 2)
    cost.append(cst)
    us_star.append(u_star)

us_star = np.array(us_star).reshape(len(us_star))
cost = np.array(cost).reshape(len(cost))
res = np.array((data[:, -1], np.array(us_star), np.array(cost))).T
