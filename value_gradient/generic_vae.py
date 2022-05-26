import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
from random import randint, random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from collections import namedtuple

__attr_names = ["in_dim", "out_dim", "e_hl_dim", "d_hl_dim", "latent_dim"]
LayerInfo = namedtuple("LayerInfo", __attr_names)


class CVAE(tf.keras.Model):
    """Conditional variational autoencoder."""

    def __init__(self, layer_inf: LayerInfo):
        super(CVAE, self).__init__()
        self.layer_inf = layer_inf
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer((self.layer_inf.in_dim[0], self.layer_inf.in_dim[1])),
                tf.keras.layers.Dense(units=self.layer_inf.e_hl_dim[0], activation=tf.nn.relu),
                tf.keras.layers.Dropout(rate=0.5),
                tf.keras.layers.Dense(units=self.layer_inf.e_hl_dim[1], activation=tf.nn.relu),
                # latent space layer
                tf.keras.layers.Dense(units=self.layer_inf.latent_dim, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=self.layer_inf.latent_dim, activation=tf.nn.relu),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.layer_inf.latent_dim,)),
                tf.keras.layers.Dense(self.layer_inf.d_hl_dim[0], activation=tf.nn.relu),
                tf.keras.layers.Dropout(rate=0.5),
                tf.keras.layers.Dense(self.layer_inf.d_hl_dim[1], activation=tf.nn.relu),
                tf.keras.layers.Dense(self.layer_inf.out_dim)
            ]
        )


    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(self.layer_inf.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    recon = tf.keras.losses.mean_squared_error()
    rl = recon(x, x_logit, sample_weight = [1, 1, 1, 0.5, 0.5, 0.5])
    kl_loss = 10 ** -4 * 2 * tf.reduce_sum(tf.exp(logvar) + mean ** 2 - 1. - logvar, 1)
    return tf.reduce_mean(kl_loss + rl)


@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.
    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


if __name__ == "__main__":
    # Constants
    epochs, it, numTrain, mb_size, dim, dimW = 1000, 0, 7000, 256, 6, 3
    dataElements = dim + 3 * 3 + 2 * dim

    # Load the data
    parent_path = "../../LearnedSamplingDistributions/"
    occ_grid = np.load(parent_path + "occ_grid.npy")
    XC = pd.read_csv(parent_path + "narrowDataFile.txt", sep=',', header=None).to_numpy()
    XC = np.delete(XC, -1, axis=1)
    X_train = XC[0:numTrain, 0:dim]
    C_train = XC[0:numTrain, dim:dataElements]
    X_test = XC[numTrain:XC.shape[0], 0:dim]
    C_test = XC[numTrain:XC.shape[0], dim:dataElements]

    cs = np.concatenate((XC[0:XC.shape[0], dim + 3 * dimW:dataElements], occ_grid), axis=1)  # occ, init, goal
    c_dim = cs.shape[1]
    c_gapsInitGoal = C_test
    C_train = cs[0:numTrain, :]
    C_test = cs[numTrain:XC.shape[0], :]

    # Build model
    layers_size = [[None, 139], 6, [512, 512], [512, 512], 3]
    model = CVAE(LayerInfo(*layers_size))

    # Data pipeline
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, C_train))
    train_dataset = train_dataset.batch(256)

    # Train net
    optimizer = tf.keras.optimizers.Adam(1e-4)
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for xc in train_dataset:
            train_step(model, train_x, optimizer)
        end_time = time.time()
