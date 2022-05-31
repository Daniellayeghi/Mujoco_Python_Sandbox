import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import numpy as np
import time
from random import randint, random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import namedtuple

__attr_names = ["in_e_dim", "in_d_dim", "out_dim", "e_hl_dim", "d_hl_dim", "latent_dim"]
LayerInfo = namedtuple("LayerInfo", __attr_names)
mse = tf.keras.losses.MeanSquaredError()
tf.config.run_functions_eagerly(False)


class CVAE(tf.keras.Model):
    """Conditional variational autoencoder."""

    def __init__(self, layer_inf: LayerInfo):
        super(CVAE, self).__init__()
        self.layer_inf = layer_inf
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer((self.layer_inf.in_e_dim,)),
                tf.keras.layers.Dense(units=self.layer_inf.e_hl_dim[0], activation=tf.nn.relu),
                tf.keras.layers.Dropout(rate=0.5),
                tf.keras.layers.Dense(units=self.layer_inf.e_hl_dim[1], activation=tf.nn.relu),
                # latent space layer
                tf.keras.layers.Dense(units=self.layer_inf.latent_dim * 2),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.layer_inf.in_d_dim,)),
                tf.keras.layers.Dense(self.layer_inf.d_hl_dim[0], activation=tf.nn.relu),
                tf.keras.layers.Dropout(rate=0.5),
                tf.keras.layers.Dense(self.layer_inf.d_hl_dim[1], activation=tf.nn.relu),
                tf.keras.layers.Dense(self.layer_inf.out_dim)
            ]
        )


    @tf.function
    def sample(self, eps=None, c=None):

        if eps is None or c is None:
            raise "Conditions and latent input are missing!"

        in_p = tf.concat(axis=1, values=[eps, c])
        return self.decode(in_p, apply_sigmoid=True)

    @tf.function
    def encode(self, in_q):
        z = self.encoder(in_q)
        mean, logvar = tf.split(z, num_or_size_splits=2, axis=1)
        return mean, logvar
    
    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def decode(self, in_p, apply_sigmoid=False):
        logits = self.decoder(in_p)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


@tf.function
def compute_loss(model, in_q):
    x_in = in_q[:, 0:dim]
    c_in = in_q[:, dim:]
    mean, logvar = model.encode(in_q)
    z = model.reparameterize(mean, logvar)
    in_p = tf.concat(axis=1, values=[z, c_in])
    x_logit = model.decode(in_p)
    recon = mse(x_in, x_logit)
    kl_loss = 10 ** -4 * 2 * tf.reduce_sum(tf.exp(logvar) + mean ** 2 - 1. - logvar, 1)
    return tf.reduce_mean(kl_loss + recon)


@tf.function
def train_step(model, in_q, optimizer, epoch):
    """Executes one training step and returns the loss.
    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, in_q)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if epoch % 10 == 0:
        tf.print("Loss:", loss, "Epoch:", epoch)

        
if __name__ == "__main__":
    # Constants
    epochs, it, numTrain, mb_size, dim, dimW, gridSize = 201, 0, 7000, 256, 6, 3, 11
    dataElements = dim + 3 * 3 + 2 * dim

    # Load the data
    parent_path = "../../LearnedSamplingDistributions/"

    occ_grid = np.load(
        parent_path + "occ_grid.npy"
    )

    XC = pd.read_csv(
        parent_path + "narrowDataFile.txt", sep=',', header=None
    ).to_numpy()

    XC = np.delete(XC, -1, axis=1)
    numEntries = XC.shape[0]
    X_train = XC[0:numTrain, 0:dim]
    C_train = XC[0:numTrain, dim:dataElements]
    X_test = XC[numTrain:numEntries, 0:dim]
    C_test = XC[numTrain:numEntries, dim:dataElements]
    cs = np.concatenate(
        (XC[0:numEntries, dim + 3 * dimW:dataElements], occ_grid), axis=1
    )

    occGridSamples = np.zeros([gridSize * gridSize, 2])
    gridPointsRange = np.linspace(0, 1, num=gridSize)

    idx = 0
    for i in gridPointsRange:
        for j in gridPointsRange:
            occGridSamples[idx, 0] = i
            occGridSamples[idx, 1] = j
            idx += 1

    c_dim = cs.shape[1]
    c_gapsInitGoal = C_test
    C_train = cs[0:numTrain, :]
    C_test = cs[numTrain:numEntries, :]

    # Build model
    layers_size = [139, 136, 6, [512, 512], [512, 512], 3]
    li = LayerInfo(*layers_size)
    model = CVAE(li)

    # Data pipeline
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    C_train = tf.convert_to_tensor(C_train, dtype=tf.float32)
    XC_train = tf.concat(axis=1, values=[X_train, C_train])
    train_dataset = tf.data.Dataset.from_tensor_slices((XC_train))
    train_dataset = train_dataset.batch(mb_size)
    train_dataset = train_dataset.shuffle(XC_train.shape[0])

    # Train net
    optimizer = tf.keras.optimizers.Adam(1e-4)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for in_q in train_dataset:
            train_step(model, in_q, optimizer, epoch)
        end_time = time.time()

    # Sample latent space
    num_viz, numTest = 3000, X_test.shape[0]
    vizIdx = np.load(parent_path + "vizIdx.npy")
    c_sample_seed = C_test[vizIdx, :]
    c_sample = np.repeat([c_sample_seed], num_viz, axis=0)
    c_viz = c_gapsInitGoal[vizIdx, :]

    y_viz = model.sample(np.random.randn(num_viz, li.latent_dim), c_sample)

    fig1 = plt.figure(figsize=(10, 6), dpi=80)
    ax1 = fig1.add_subplot(111, aspect='equal')

    plt.scatter(y_viz[:, 0], y_viz[:, 1], color="green", s=70, alpha=0.1)

    dw, dimW = 0.1, 3
    gap1 = c_viz[0:3]
    gap2 = c_viz[3:6]
    gap3 = c_viz[6:9]
    init = c_viz[9:15]
    goal = c_viz[15:21]

    obs1 = [0, gap1[1] - dw, -0.5, gap1[0], gap1[1], 1.5]
    obs2 = [gap2[0] - dw, 0, -0.5, gap2[0], gap2[1], 1.5]
    obs3 = [gap2[0] - dw, gap2[1] + dw, -0.5, gap2[0], 1, 1.5]
    obs4 = [gap1[0] + dw, gap1[1] - dw, -0.5, gap3[0], gap1[1], 1.5]
    obs5 = [gap3[0] + dw, gap1[1] - dw, -0.5, 1, gap1[1], 1.5]
    obsBounds = [-0.1, -0.1, -0.5, 0, 1.1, 1.5,
                 -0.1, -0.1, -0.5, 1.1, 0, 1.5,
                 -0.1, 1, -0.5, 1.1, 1.1, 1.5,
                 1, -0.1, -0.5, 1.1, 1.1, 1.5, ]


    obs = np.concatenate((obs1, obs2, obs3, obs4, obs5, obsBounds), axis=0)
    for i in range(0, int(obs.shape[0] / (2 * dimW))):
        ax1.add_patch(
            patches.Rectangle(
                (obs[i * 2 * dimW], obs[i * 2 * dimW + 1]),  # (x,y)
                obs[i * 2 * dimW + dimW] - obs[i * 2 * dimW],  # width
                obs[i * 2 * dimW + dimW + 1] - obs[i * 2 * dimW + 1],  # height
                alpha=0.6
            ))

    for i in range(0, gridSize * gridSize):  # plot occupancy grid
        cIdx = i + 2 * dim
        if c_sample_seed[cIdx] == 0:
            plt.scatter(occGridSamples[i, 0], occGridSamples[i, 1], color="red", s=50, alpha=0.7)
        else:
            plt.scatter(occGridSamples[i, 0], occGridSamples[i, 1], color="green", s=50, alpha=0.7)

    plt.scatter(init[0], init[1], color="red", s=250, edgecolors='black')  # init
    plt.scatter(goal[0], goal[1], color="blue", s=250, edgecolors='black')  # goal

    plt.show()

    plt.figure(figsize=(10, 6), dpi=80)
    viz1, viz2 = 1, 4
    plt.scatter(y_viz[:, viz1], y_viz[:, viz2], color="green", s=70, alpha=0.1)
    plt.scatter(c_viz[viz1 + 9], c_viz[viz2 + 9], color="red", s=250, edgecolors='black')  # init
    plt.scatter(c_viz[viz1 + 9 + dim], c_viz[viz2 + 9 + dim], color="blue", s=500, edgecolors='black')  # goal
    plt.show()
