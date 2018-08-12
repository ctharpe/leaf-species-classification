# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# ==============================================================================
#
# File modified by Christopher S. Tharpe, February - August, 2017
#
# implements a convolutional neural network (CNN) augmented with numerical
# input data
#
# ==============================================================================


"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

from data_input import LeafData
from data_input import NUM_CLASSES

#setup numpy print options to show entire vectors/tensors, not ellipses:
import numpy
numpy.set_printoptions(threshold=numpy.nan)

import argparse
import sys
import time
import math
import random

from data_input import LeafData
from data_input import NUM_CLASSES

import tensorflow as tf

log_path = 'log_files/'

FLAGS = None

#Image size, in pixels, assumes square image:

IMAGE_SIZE = 32

NUM_COLOR_CHANNELS = 1

PATCH_SIZE = 5

NUM_LAYER_1_FEATURES = 16

NUM_LAYER_2_FEATURES = 32

NUM_FULLY_CONNECTED_NEURONS = 512

NUM_STATISTICAL_PROPERTIES = 192

BATCH_SIZE = 99

NUM_TRAINING_EPOCHS = 10000

MIN_ANGLE = -15
MAX_ANGLE = 15

def weight_variable(shape, variable_name):
  initial = tf.truncated_normal(shape, stddev=0.1, name = variable_name)
  return tf.Variable(initial)

def bias_variable(shape, variable_name):
  initial = tf.constant(0.1, shape=shape, name = variable_name)
  return tf.Variable(initial)



def conv2d(x, W, variable_name):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name = variable_name)


def max_pool_2x2(x, variable_name):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def get_rotation_angles():
  list_of_angles = []

  for i in range(0, BATCH_SIZE):
    list_of_angles.append(math.radians(random.randrange(MIN_ANGLE, MAX_ANGLE)))

  return list_of_angles


def get_list_of_zeros(length):

  list_of_zeros = [0.0] * length

  return list_of_zeros


def main(_):
  # Import data

  leaf_data = LeafData()
  leaf_data.import_data()

  print("tensorflow version = ", tf.__version__)

  # Create the model

  x = tf.placeholder(tf.float32, [None, IMAGE_SIZE * IMAGE_SIZE], name = "IMG_IN")


  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

  with tf.name_scope('CONV_1') as scope:

    W_conv1 = weight_variable([PATCH_SIZE, PATCH_SIZE, NUM_COLOR_CHANNELS, NUM_LAYER_1_FEATURES], "WEIGHT_CONV_1")
    b_conv1 = bias_variable([NUM_LAYER_1_FEATURES], "BIAS_CONV_1")


    x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_COLOR_CHANNELS], name = "RESHAPED_IMAGE_INPUT")

    rotation_angles = tf.placeholder(tf.float32, [None], name = "ROTATION_ANGLES")

    x_image = tf.contrib.image.rotate(x_image, rotation_angles)

    h_conv1 = tf.nn.tanh(conv2d(x_image, W_conv1, "CONV_2D_1") + b_conv1, name="LAYER_1_OUTPUT")


  with tf.name_scope('POOLING_1') as scope:
    h_pool1 = max_pool_2x2(h_conv1, "POOLED_LAYER_1_OUTPUT")


  with tf.name_scope('CONV_2') as scope:
    W_conv2 = weight_variable([PATCH_SIZE, PATCH_SIZE, NUM_LAYER_1_FEATURES, NUM_LAYER_2_FEATURES], "WEIGHT_CONV_2")
    b_conv2 = bias_variable([NUM_LAYER_2_FEATURES], "BIAS_CONV_2")

    h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2, "CONV_2D_2") + b_conv2, name = "LAYER_2_OUTPUT")

  with tf.name_scope('POOLING_2') as scope:
    h_pool2 = max_pool_2x2(h_conv2, "POOLED_LAYER_2_OUTPUT")

  with tf.name_scope('F_C_1') as scope:

    reduced_size = int(IMAGE_SIZE / 4)

    W_fc1 = weight_variable([(reduced_size * reduced_size * NUM_LAYER_2_FEATURES) + NUM_STATISTICAL_PROPERTIES, NUM_FULLY_CONNECTED_NEURONS], "WEIGHT_FULLY_CONN_1")
    b_fc1 = bias_variable([NUM_FULLY_CONNECTED_NEURONS], "BIAS_FULLY_CONN_1")

    temp = tf.reshape(h_pool2, [-1, reduced_size * reduced_size * NUM_LAYER_2_FEATURES])

    paddings = [[0, 0],
                [0, NUM_STATISTICAL_PROPERTIES]]

    temp_padded = tf.pad(temp, paddings, 'CONSTANT')

    h_pool2_flat = tf.reshape(temp_padded,
                              [-1, reduced_size * reduced_size * NUM_LAYER_2_FEATURES + NUM_STATISTICAL_PROPERTIES])

  with tf.name_scope('NUM_IN') as scope:

    # add augmentation data here
    x_statistics = tf.placeholder(tf.float32, [None, NUM_STATISTICAL_PROPERTIES])
    temp_statistics = tf.reshape(x_statistics, [-1, NUM_STATISTICAL_PROPERTIES])

  with tf.name_scope('COMBINED') as scope:

    paddings = [[0, 0],
                [reduced_size * reduced_size * NUM_LAYER_2_FEATURES, 0]]

    temp_statistics_padded = tf.pad(temp_statistics, paddings, 'CONSTANT', name="statistics_padded")

    h_pool2_flat = h_pool2_flat + temp_statistics_padded

    h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1, name = "HP2_WFC1") + b_fc1, "LAYER_3_OUTPUT")

  with tf.name_scope('F_C_2') as scope:

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([NUM_FULLY_CONNECTED_NEURONS, NUM_CLASSES], "WEIGHT_FULLY_CONN_2")
    b_fc2 = bias_variable([NUM_CLASSES], "BIAS_FULLY_CONN_2")

  with tf.name_scope('OUTPUT') as scope:

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


  cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  t1 = time.time()

  train_accuracy = 0.0

  for i in range(NUM_TRAINING_EPOCHS):

    batch_xs, batch_ys, batch_stats = leaf_data.get_next_batch(BATCH_SIZE)

    list_of_angles = get_rotation_angles()

    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
        x: batch_xs, y_: batch_ys, x_statistics: batch_stats, rotation_angles: list_of_angles, keep_prob: 1.0})
      print("step %d, training accuracy %g" % (i, train_accuracy))

    train_step.run(
      feed_dict={x: batch_xs, y_: batch_ys, x_statistics: batch_stats, rotation_angles: list_of_angles, keep_prob: 0.5})

  t2 = time.time()

  # Test trained model

  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  print("training time (in seconds) =", t2-t1)

  #WRITE SUMMARIES
  tf.summary.merge_all()
  file_writer = tf.summary.FileWriter(log_path, sess.graph)

  list_of_angles = get_list_of_zeros(len(leaf_data.train_images))

  print("train accuracy %g" % accuracy.eval(feed_dict={
    x: leaf_data.train_images, y_: leaf_data.train_labels, rotation_angles: list_of_angles,
    x_statistics: leaf_data.train_statistics, keep_prob: 1.0}))

  prediction = tf.nn.softmax(y_conv)

  list_of_angles = get_list_of_zeros(len(leaf_data.test_images))

  test_results = prediction.eval(
    feed_dict={x: leaf_data.test_images, x_statistics: leaf_data.test_statistics, rotation_angles: list_of_angles,
               keep_prob: 1.0})
  leaf_data.create_output_file(test_results, "x_stats_output.csv")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--data_dir', type=str, default='/home/user/UD/LEAF/train_file_abbreviated',
                                          help='Directory for storing input data')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


