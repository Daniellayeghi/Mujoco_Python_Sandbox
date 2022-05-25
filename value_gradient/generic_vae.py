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
__attr_vals = [[None, 139], 6, [512, 512], [512, 512], 3]
LayerInfo = namedtuple("LayerInfo", __attr_names)
LayerInfo(*__attr_vals)


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
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


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
    epochs, it, numTrain, mb_size, dim = 1000, 0, 7000, 256, 6
    dataElements = dim + 3 * 3 + 2 * dim
    # Load the data
    occ_grid = np.load("../../LearnedSamplingDistributions/occ_grid.npy")
    XC = pd.read_csv("./../../LearnedSamplingDistributions/narrowDataFile.txt", sep=',').to_numpy()
    XC = np.delete(XC, -1, axis=1)
    X = XC[0:numTrain, 0:dim]
    C = XC[0:numTrain, dim:dataElements]
    # Batch the data to combine the consecutive elements into batches

    # Model setup
    model = CVAE(LayerInfo(*__attr_vals))


    # for epoch in range(1, epochs + 1):
    #     start_time = time.time()
    #     for train_x in train_dataset:
    #         train_step(model, train_x, optimizer)
    #     end_time = time.time()
    #
    #     loss = tf.keras.metrics.Mean()
    #     for test_x in test_dataset:
    #         loss(compute_loss(model, test_x))
    #     elbo = -loss.result()
    #     display.clear_output(wait=False)
    #     print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
    #           .format(epoch, elbo, end_time - start_time))
    #
    # for it in range(it, it + epochs):
    #     # randomly generate batches
    #     batch_elements = [randint(0, numTrain - 1) for n in range(0, mb_size)]
    #     X_mb = X_train[batch_elements, :]
    #     c_mb = c_train[batch_elements, :]
    #
    #     _, loss = sess.run([train_step, cvae_loss], feed_dict={X: X_mb, c: c_mb})
    #
    #     if it % 1000 == 0:
    #         print('Iter: {}'.format(it))
    #         print('Loss: {:.4}'.format(loss))
    #         print()