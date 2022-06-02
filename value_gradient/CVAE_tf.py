import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import numpy as np

from collections import namedtuple

__attr_names = ["in_e_dim", "in_d_dim", "out_dim", "e_hl_dim", "d_hl_dim", "latent_dim"]
LayerInfo = namedtuple("LayerInfo", __attr_names)
mse = tf.keras.losses.MeanSquaredError()
tf.config.run_functions_eagerly(True)


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
