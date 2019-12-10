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
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import \
    BatchNormalization, Input, Dense, Reshape, Flatten, Embedding, multiply

from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam


def build_generator(z_input: Input, label_input: Input):
    """
    Build generator CNN
    :param z_input: latent input
    :param label_input: conditional label input
    """

    model = Sequential([
        Dense(128, input_dim=latent_dim),
        LeakyReLU(alpha=0.2), BatchNormalization(momentum=0.8),
        Dense(256),
        LeakyReLU(alpha=0.2), BatchNormalization(momentum=0.8),
        Dense(512),
        LeakyReLU(alpha=0.2), BatchNormalization(momentum=0.8),
        Dense(np.prod((28, 28, 1)), activation='tanh'),
        # reshape to MNIST image size
        Reshape((28, 28, 1))
    ])

    model.summary()

    # the latent input vector z
    label_embedding = Embedding(input_dim=10, output_dim=latent_dim)(label_input)
    flat_embedding = Flatten()(label_embedding)

    # combine the noise and label by element-wise multiplication
    model_input = multiply([z_input, flat_embedding])
    image = model(model_input)

    return Model([z_input, label_input], image)


def build_discriminator():
    """
    Build discriminator network
    """

    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(128),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid'),
    ], name='discriminator')

    model.summary()

    image = Input(shape=(28, 28, 1))
    flat_img = Flatten()(image)

    label_input = Input(shape=(1,), dtype='int32')
    label_embedding = Embedding(input_dim=10, output_dim=np.prod((28, 28, 1)))(label_input)
    flat_embedding = Flatten()(label_embedding)

    # combine the noise and label by element-wise multiplication
    model_input = multiply([flat_img, flat_embedding])

    validity = model(model_input)

    return Model([image, label_input], validity)


def train(generator, discriminator, combined, steps, batch_size):
    """
    Train the GAN system
    :param generator: generator
    :param discriminator: discriminator
    :param combined: stacked generator and discriminator
    we'll use the combined network when we train the generator
    :param steps: number of alternating steps for training
    :param batch_size: size of the minibatch
    """

    # Load the dataset
    (x_train, x_labels), _ = mnist.load_data()

    # Rescale in [-1, 1] interval
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=-1)

    # Discriminator ground truths
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for step in range(steps):
        # Train the discriminator
        # Select a random batch of images and labels
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_images, labels = x_train[idx], x_labels[idx]

        # Random batch of noise
        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Generate a batch of new images
        generated_images = generator.predict([noise, labels])

        # Train the discriminator
        discriminator_real_loss = discriminator.train_on_batch([real_images, labels], real)
        discriminator_fake_loss = discriminator.train_on_batch([generated_images, labels], fake)
        discriminator_loss = 0.5 * np.add(discriminator_real_loss, discriminator_fake_loss)

        # Train the generator
        # random latent vector z
        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Condition on labels
        sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

        # Train the generator
        # Note that we use the "valid" labels for the generated images
        # That's because we try to maximize the discriminator loss
        generator_loss = combined.train_on_batch([noise, sampled_labels], real)

        # Display progress
        print("%d [Discriminator loss: %.4f%%, acc.: %.2f%%] [Generator loss: %.4f%%]" %
              (step, discriminator_loss[0], 100 * discriminator_loss[1], generator_loss))


def plot_generated_images(generator, label: int):
    """
    Display a nxn 2D manifold of digits
    :param generator: the generator
    :param label: generate images of particular label
    """
    n = 10
    digit_size = 28

    # big array containing all images
    figure = np.zeros((digit_size * n, digit_size * n))

    # n*n random latent distributions
    noise = np.random.normal(0, 1, (n * n, latent_dim))
    sampled_labels = np.full(n * n, label, dtype=np.int64).reshape(-1, 1)

    # generate the images
    generated_images = generator.predict([noise, sampled_labels])

    # fill the big array with images
    for i in range(n):
        for j in range(n):
            slice_i = slice(i * digit_size, (i + 1) * digit_size)
            slice_j = slice(j * digit_size, (j + 1) * digit_size)
            figure[slice_i, slice_j] = np.reshape(generated_images[i * n + j], (28, 28))

    # plot the results
    plt.figure(num=label, figsize=(6, 5))
    plt.axis('off')
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
    plt.close()


if __name__ == '__main__':
    print("CGAN for new MNIST images with Keras")

    latent_dim = 64

    # we'll use Adam optimizer
    optimizer = Adam(0.0002, 0.5)

    # Build and compile the discriminator
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

    # Build the generator
    z = Input(shape=(latent_dim,))
    label = Input(shape=(1,))

    generator = build_generator(z, label)

    # Generator input z
    generated_image = generator([z, label])

    # Only train the generator for the combined model
    discriminator.trainable = False

    # The discriminator takes generated image as input and determines validity
    real_or_fake = discriminator([generated_image, label])

    # Stack the generator and discriminator in a combined model
    # Trains the generator to deceive the discriminator
    combined = Model([z, label], real_or_fake)
    combined.compile(loss='binary_crossentropy',
                     optimizer=optimizer)

    # train the GAN system
    train(generator=generator,
          discriminator=discriminator,
          combined=combined,
          steps=50000,
          batch_size=100)

    # display some random generated images
    plot_generated_images(generator, 1)
    plot_generated_images(generator, 2)
    plot_generated_images(generator, 3)
    plot_generated_images(generator, 4)
    plot_generated_images(generator, 5)
    plot_generated_images(generator, 6)
    plot_generated_images(generator, 7)
    plot_generated_images(generator, 8)
    plot_generated_images(generator, 9)
    plot_generated_images(generator, 0)
