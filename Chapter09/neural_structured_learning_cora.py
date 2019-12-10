"""
This example is partially based on https://www.tensorflow.org/neural_structured_learning/tutorials/graph_keras_mlp_cora
License of the base version: Apache License, Version 2.0
"""

import neural_structured_learning as nsl
import tensorflow as tf

# Cora dataset path
TRAIN_DATA_PATH = 'data/train_merged_examples.tfr'
TEST_DATA_PATH = 'data/test_examples.tfr'
# Constants used to identify neighbor features in the input.
NBR_FEATURE_PREFIX = 'NL_nbr_'
NBR_WEIGHT_SUFFIX = '_weight'
# Dataset parameters
NUM_CLASSES = 7
MAX_SEQ_LENGTH = 1433
# Number of neighbors to consider in the composite loss function
NUM_NEIGHBORS = 1
# Training parameters
BATCH_SIZE = 128


def parse_example(example_proto: tf.train.Example) -> tuple:
    """
    Extracts relevant fields from the example_proto
    :param example_proto: the example
    :return: A data/label pair, where
        data is a dictionary containing relevant features of the example
    """
    # The 'words' feature is a multi-hot, bag-of-words representation of the
    # original raw text. A default value is required for examples that don't
    # have the feature.
    feature_spec = {
        'words':
            tf.io.FixedLenFeature(shape=[MAX_SEQ_LENGTH],
                                  dtype=tf.int64,
                                  default_value=tf.constant(
                                      value=0,
                                      dtype=tf.int64,
                                      shape=[MAX_SEQ_LENGTH])),
        'label':
            tf.io.FixedLenFeature((), tf.int64, default_value=-1),
    }
    # We also extract corresponding neighbor features in a similar manner to
    # the features above.
    for i in range(NUM_NEIGHBORS):
        nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'words')
        nbr_weight_key = '{}{}{}'.format(NBR_FEATURE_PREFIX, i, NBR_WEIGHT_SUFFIX)
        feature_spec[nbr_feature_key] = tf.io.FixedLenFeature(
            shape=[MAX_SEQ_LENGTH],
            dtype=tf.int64,
            default_value=tf.constant(
                value=0, dtype=tf.int64, shape=[MAX_SEQ_LENGTH]))

        # We assign a default value of 0.0 for the neighbor weight so that
        # graph regularization is done on samples based on their exact number
        # of neighbors. In other words, non-existent neighbors are discounted.
        feature_spec[nbr_weight_key] = tf.io.FixedLenFeature(
            shape=[1], dtype=tf.float32, default_value=tf.constant([0.0]))

    features = tf.io.parse_single_example(example_proto, feature_spec)

    labels = features.pop('label')
    return features, labels


def make_dataset(file_path: str, training=False) -> tf.data.TFRecordDataset:
    """
    Extracts relevant fields from the example_proto
    :param file_path: name of the file in the `.tfrecord` format containing
        `tf.train.Example` objects.
    :param training: whether the dataset is for training
    :return: tf.data.TFRecordDataset of tf.train.Example objects
    """
    dataset = tf.data.TFRecordDataset([file_path])
    if training:
        dataset = dataset.shuffle(10000)
    dataset = dataset.map(parse_example).batch(BATCH_SIZE)

    return dataset


train_dataset = make_dataset(TRAIN_DATA_PATH, training=True)
test_dataset = make_dataset(TEST_DATA_PATH)


def build_model(dropout_rate):
    """Creates a sequential multi-layer perceptron model."""
    return tf.keras.Sequential([
        # one-hot encoded input.
        tf.keras.layers.InputLayer(
            input_shape=(MAX_SEQ_LENGTH,), name='words'),

        # 2 fully-connected layers + dropout
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),

        # Softmax output
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])


# Build a new base MLP model.
model = build_model(dropout_rate=0.5)

# Wrap the base MLP model with graph regularization.
graph_reg_config = nsl.configs.make_graph_reg_config(
    max_neighbors=NUM_NEIGHBORS,
    multiplier=0.1,
    distance_type=nsl.configs.DistanceType.L2,
    sum_over_axis=-1)
graph_reg_model = nsl.keras.GraphRegularization(model,
                                                graph_reg_config)
graph_reg_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# run eagerly to prevent epoch warnings
graph_reg_model.run_eagerly = True

graph_reg_model.fit(train_dataset, epochs=100, verbose=1)

eval_results = dict(
    zip(graph_reg_model.metrics_names,
        graph_reg_model.evaluate(test_dataset)))
print('Evaluation accuracy: {}'.format(eval_results['accuracy']))
print('Evaluation loss: {}'.format(eval_results['loss']))
print('Evaluation graph loss: {}'.format(eval_results['graph_loss']))
