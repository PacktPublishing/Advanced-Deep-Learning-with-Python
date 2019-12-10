"""
MIT License

This example is based on https://github.com/eriklindernoren/Keras-GAN
Copyright (c) 2017 Erik Linder-Norén
Copyright (c) 2019 Ivan Vasilev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import \
    BatchNormalization, Input, Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop


def build_generator(latent_input: Input):
    """
    Build generator FC network
    :param latent_input: the latent input
    """

    model = Sequential([
        # start with a fully connected layer to upsample the 1d latent vector
        # the input_shape is the same as latent_input (excluding the mini-batch)

        Dense(64, input_shape=latent_input.shape[1:]),
        BatchNormalization(),
        LeakyReLU(0.2),

        Dense(128),
        BatchNormalization(),
        LeakyReLU(0.2),

        Dense(256),
        BatchNormalization(),
        LeakyReLU(0.2),

        Dense(512),
        BatchNormalization(),
        LeakyReLU(0.2),

        Dense(28 * 28, activation='tanh'),
    ])

    model.summary()

    # this is forward phase
    generated = model(latent_input)

    # build model from the input and output
    return Model(z, generated)


def build_critic():
    """
    Build critic FC network
    """

    model = Sequential([
        Dense(512, input_shape=(28 * 28,)),
        LeakyReLU(0.2),
        Dense(256),
        LeakyReLU(0.2),
        Dense(1),
    ])

    model.summary()

    image = Input(shape=(28 * 28,))
    output = model(image)

    return Model(image, output)


def train(generator, critic, combined, steps, batch_size, n_critic, clip_value):
    """
    Train the GAN model
    :param generator: generator model
    :param critic: the critic model
    :param combined: stacked generator and critic
        we'll use the combined network when we train the generator
    :param steps: number of alternating steps for training
    :param batch_size: size of the minibatch
    :param n_critic: how many critic training steps for one generator step
    :param clip_value: clip value for the critic weights
    """

    # Load the dataset
    (x_train, _), _ = mnist.load_data()

    # Rescale in [-1, 1] interval
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5

    # We use FC networks, so we flatten the array
    x_train = x_train.reshape(x_train.shape[0], 28 * 28)

    # Discriminator ground truths
    real = np.ones((batch_size, 1))
    fake = -np.ones((batch_size, 1))

    latent_dim = generator.input_shape[1]

    # Train for number of steps
    for step in range(steps):
        # Train the critic first for n_critic steps
        for _ in range(n_critic):
            # Select a random batch of images
            real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, latent_dim))

            # Generate a batch of new images
            generated_images = generator.predict(noise)

            # Train the critic
            critic_real_loss = critic.train_on_batch(real_images, real)
            critic_fake_loss = critic.train_on_batch(generated_images, fake)
            critic_loss = 0.5 * np.add(critic_real_loss, critic_fake_loss)

            # Clip critic weights
            for l in critic.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                l.set_weights(weights)

        # Train the generator
        # Note that we use the "valid" labels for the generated images
        # That's because we try to maximize the discriminator loss
        generator_loss = combined.train_on_batch(noise, real)

        # Display progress
        print("%d [Critic loss: %.4f%%] [Generator loss: %.4f%%]" %
              (step, critic_loss[0], generator_loss))


def plot_generated_images(generator):
    """
    Display a nxn 2D manifold of digits
    :param generator: the generator
    """
    n = 10
    digit_size = 28

    # big array containing all images
    figure = np.zeros((digit_size * n, digit_size * n))

    latent_dim = generator.input_shape[1]

    # n*n random latent distributions
    noise = np.random.normal(0, 1, (n * n, latent_dim))

    # generate the images
    generated_images = generator.predict(noise)

    # fill the big array with images
    for i in range(n):
        for j in range(n):
            slice_i = slice(i * digit_size, (i + 1) * digit_size)
            slice_j = slice(j * digit_size, (j + 1) * digit_size)
            figure[slice_i, slice_j] = np.reshape(generated_images[i * n + j], (28, 28))

    # plot the results
    plt.figure(figsize=(6, 5))
    plt.axis('off')
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


def wasserstein_loss(y_true, y_pred):
    """The Wasserstein loss implementation"""
    return tensorflow.keras.backend.mean(y_true * y_pred)


if __name__ == '__main__':
    print("Wasserstein GAN for new MNIST images with TF/Keras")

    latent_dim = 100

    # Build the generator
    # Generator input z
    z = Input(shape=(latent_dim,))

    generator = build_generator(z)

    generated_image = generator(z)

    # we'll use RMSprop optimizer
    optimizer = RMSprop(lr=0.00005)

    # Build and compile the discriminator
    critic = build_critic()
    critic.compile(optimizer, wasserstein_loss,
                   metrics=['accuracy'])

    # The discriminator takes generated image as input and determines validity
    real_or_fake = critic(generated_image)

    # Only train the generator for the combined model
    critic.trainable = False

    # Stack the generator and discriminator in a combined model
    # Trains the generator to deceive the discriminator
    combined = Model(z, real_or_fake)
    combined.compile(loss=wasserstein_loss, optimizer=optimizer)

    # train the GAN system
    train(generator, critic, combined,
          steps=40000, batch_size=100, n_critic=5, clip_value=0.01)

    # display some random generated images
    plot_generated_images(generator)
