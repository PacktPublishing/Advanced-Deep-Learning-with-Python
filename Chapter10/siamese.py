"""
COPYRIGHT

All contributions by François Chollet:
Copyright (c) 2015 - 2019, François Chollet.
All rights reserved.

All contributions by Google:
Copyright (c) 2015 - 2019, Google, Inc.
All rights reserved.

All contributions by Microsoft:
Copyright (c) 2017 - 2019, Microsoft, Inc.
All rights reserved.

All other contributions:
Copyright (c) 2015 - 2019, the respective contributors.
All rights reserved.

Copyright (c) 2015 - 2019, Ivan Vasilev.
This example is partially based on https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py

Each contributor holds copyright over their respective contributions.
The project versioning (Git) records all such contribution source information.

LICENSE

The MIT License (MIT)

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
import random

import numpy as np
import tensorflow as tf


def create_pairs(inputs: np.ndarray, labels: np.ndarray):
    """Create equal number of true/false pairs of samples"""

    num_classes = 10

    digit_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    pairs = list()
    labels = list()
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[inputs[z1], inputs[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[inputs[z1], inputs[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels, dtype=np.float32)


def create_base_network():
    """The shared encoding part of the siamese network"""

    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64, activation='relu'),
    ])


if __name__ == '__main__':
    # Load the train and test MNIST datasets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_train /= 255
    x_test /= 255
    input_shape = x_train.shape[1:]

    # Create true/false training and testing pairs
    train_pairs, tr_labels = create_pairs(x_train, y_train)
    test_pairs, test_labels = create_pairs(x_test, y_test)

    # Create the siamese network
    # Start from the shared layers
    base_network = create_base_network()

    # Create first half of the siamese system
    input_a = tf.keras.layers.Input(shape=input_shape)

    # Note how we reuse the base_network in both halfs
    encoder_a = base_network(input_a)

    # Create the second half of the siamese system
    input_b = tf.keras.layers.Input(shape=input_shape)
    encoder_b = base_network(input_b)

    # Create the the distance measure
    l1_dist = tf.keras.layers.Lambda(
        lambda embeddings: tf.keras.backend.abs(embeddings[0] - embeddings[1])) \
        ([encoder_a, encoder_b])

    # Final fc layer with a single logistic output for the binary classification
    flattened_weighted_distance = tf.keras.layers.Dense(1, activation='sigmoid') \
        (l1_dist)

    # Build the model
    model = tf.keras.models.Model([input_a, input_b], flattened_weighted_distance)

    # Train
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit([train_pairs[:, 0], train_pairs[:, 1]], tr_labels,
              batch_size=128,
              epochs=20,
              validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_labels))
