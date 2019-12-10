"""
This example is partially based on https://github.com/tensorflow/docs/blob/master/site/en/tutorials/images/transfer_learning.ipynb

Copyright 2018 The TensorFlow Authors.  All rights reserved.
Copyright 2019 Ivan Vasilev.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

IMG_SIZE = 224
BATCH_SIZE = 50

data, metadata = tfds.load('cifar10', with_info=True, as_supervised=True)

raw_train, raw_test = data['train'].repeat(), data['test'].repeat()


def train_format_sample(image, label):
    """Transform data for training"""
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = (image / 127.5) - 1
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    label = tf.one_hot(label, metadata.features['label'].num_classes)

    return image, label


def test_format_sample(image, label):
    """Transform data for testing"""
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = (image / 127.5) - 1

    label = tf.one_hot(label, metadata.features['label'].num_classes)

    return image, label


# assign transformers to raw data
train_data = raw_train.map(train_format_sample)
test_data = raw_test.map(test_format_sample)

# extract batches from the training set
train_batches = train_data.shuffle(1000).batch(BATCH_SIZE)
test_batches = test_data.batch(BATCH_SIZE)


def build_fe_model():
    """"Create feature extraction model from the pre-trained model ResNet50V2"""

    # create the pre-trained part of the network, excluding FC layers
    base_model = tf.keras.applications.ResNet50V2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                  include_top=False,
                                                  weights='imagenet')

    # exclude all model layers from training
    base_model.trainable = False

    # create new model as a combination of the pre-trained net
    # and one fully connected layer at the top
    return tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(
            metadata.features['label'].num_classes,
            activation='softmax')
    ])


def build_ft_model():
    """"Create fine tuning model from the pre-trained model ResNet50V2"""

    # create the pre-trained part of the network, excluding FC layers
    base_model = tf.keras.applications.ResNet50V2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                  include_top=False,
                                                  weights='imagenet')

    # Fine tune from this layer onwards
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # create new model as a combination of the pre-trained net
    # and one fully connected layer at the top
    return tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(
            metadata.features['label'].num_classes,
            activation='softmax')
    ])


def train_model(model, epochs=5):
    """Train the model. This function is shared for both FE and FT modes"""

    # configure the model for training
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # train the model
    history = model.fit(train_batches,
                        epochs=epochs,
                        steps_per_epoch=metadata.splits['train'].num_examples // BATCH_SIZE,
                        validation_data=test_batches,
                        validation_steps=metadata.splits['test'].num_examples // BATCH_SIZE,
                        workers=4)

    # plot accuracy
    test_acc = history.history['val_accuracy']

    plt.figure()
    plt.plot(test_acc)
    plt.xticks(
        [i for i in range(0, len(test_acc))],
        [i + 1 for i in range(0, len(test_acc))])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Transfer learning with feature extraction or fine tuning")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-fe', action='store_true', help="Feature extraction")
    group.add_argument('-ft', action='store_true', help="Fine tuning")
    args = parser.parse_args()

    if args.ft:
        print("Transfer learning: fine tuning with Keras ResNet50V2 network for CIFAR-10")
        model = build_ft_model()
        model.summary()
        train_model(model)
    elif args.fe:
        print("Transfer learning: feature extractor with Keras ResNet50V2 network for CIFAR-10")
        model = build_fe_model()
        model.summary()
        train_model(model)
