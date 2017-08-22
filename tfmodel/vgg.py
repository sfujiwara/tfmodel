# -*- coding: utf-8 -*-

import os
import util
import tensorflow as tf
from tensorflow.python.util.deprecation import deprecated


MODEL_URL = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
VGG_MEAN = [123.68, 116.779, 103.939]
VGG16_GRAPH_KEY = "vgg16"


def preprocess(img_tensor):
    with tf.name_scope("preprocessing"):
        preprocessed_img_tensor = img_tensor - tf.constant(VGG_MEAN, name="vgg_mean")
    return preprocessed_img_tensor


def vgg_conv2d(inputs, filters, trainable=True):
    n_kernels = inputs.get_shape()[3].value
    w = tf.get_variable(
        name="weights",
        shape=[3, 3, n_kernels, filters],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(),
        trainable=trainable,
        collections=[VGG16_GRAPH_KEY, tf.GraphKeys.GLOBAL_VARIABLES],
    )
    b = tf.get_variable(
        name="biases",
        shape=[filters],
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=trainable,
        collections=[VGG16_GRAPH_KEY, tf.GraphKeys.GLOBAL_VARIABLES],
    )
    h = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding="SAME"), b))
    return h, w, b


def vgg_fc(inputs, shape, activation_fn=None):
    w = tf.get_variable(
        name="weights",
        shape=shape,
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(),
        collections=[VGG16_GRAPH_KEY, tf.GraphKeys.GLOBAL_VARIABLES],
    )
    b = tf.get_variable(
        name="biases",
        shape=[shape[-1]],
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        collections=[VGG16_GRAPH_KEY, tf.GraphKeys.GLOBAL_VARIABLES],
    )
    h = tf.nn.bias_add(tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding="VALID"), b)
    if activation_fn:
        h = activation_fn(h)
    return h, w, b


def build_vgg16_graph(img_tensor, reuse=False, trainable=True, include_top=False):
    # img_tensor = preprocess(img_tensor)
    with tf.variable_scope("vgg_16", reuse=reuse):
        # Convolution layers 1
        with tf.variable_scope("conv1"):
            with tf.variable_scope("conv1_1"):
                h_conv1_1, w_conv1_1, b_conv1_1 = vgg_conv2d(img_tensor, 64, trainable)
            with tf.variable_scope("conv1_2"):
                h_conv1_2, w_conv1_2, b_conv1_2 = vgg_conv2d(h_conv1_1, 64, trainable)
        # Pooling 1
        pool1 = tf.nn.max_pool(
            h_conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1"
        )
        # Convolution layers 2
        with tf.variable_scope("conv2"):
            with tf.variable_scope("conv2_1"):
                h_conv2_1, w_conv2_1, b_conv2_1 = vgg_conv2d(pool1, 128, trainable)
            with tf.variable_scope("conv2_2"):
                h_conv2_2, w_conv2_2, b_conv2_2 = vgg_conv2d(h_conv2_1, 128, trainable)
        # Pooling 2
        pool2 = tf.nn.max_pool(
            h_conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2"
        )
        # Convolution layers 3
        with tf.variable_scope("conv3"):
            with tf.variable_scope("conv3_1"):
                h_conv3_1, w_conv3_1, b_conv3_1 = vgg_conv2d(pool2, 256, trainable)
            with tf.variable_scope("conv3_2"):
                h_conv3_2, w_conv3_2, b_conv3_2 = vgg_conv2d(h_conv3_1, 256, trainable)
            with tf.variable_scope("conv3_3"):
                h_conv3_3, w_conv3_3, b_conv3_3 = vgg_conv2d(h_conv3_2, 256, trainable)
        # Pooling 3
        pool3 = tf.nn.max_pool(
            h_conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool3"
        )
        # Convolution 4
        with tf.variable_scope("conv4"):
            with tf.variable_scope("conv4_1"):
                h_conv4_1, w_conv4_1, b_conv4_1 = vgg_conv2d(pool3, 512, trainable)
            with tf.variable_scope("conv4_2"):
                h_conv4_2, w_conv4_2, b_conv4_2 = vgg_conv2d(h_conv4_1, 512, trainable)
            with tf.variable_scope("conv4_3"):
                h_conv4_3, w_conv4_3, b_conv4_3 = vgg_conv2d(h_conv4_2, 512, trainable)
        # Pooling 4
        pool4 = tf.nn.max_pool(
            h_conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool4"
        )
        # Convolution 5
        with tf.variable_scope("conv5"):
            with tf.variable_scope("conv5_1"):
                h_conv5_1, w_conv5_1, b_conv5_1 = vgg_conv2d(pool4, 512, trainable)
            with tf.variable_scope("conv5_2"):
                h_conv5_2, w_conv5_2, b_conv5_2 = vgg_conv2d(h_conv5_1, 512, trainable)
            with tf.variable_scope("conv5_3"):
                h_conv5_3, w_conv5_3, b_conv5_3 = vgg_conv2d(h_conv5_2, 512, trainable)
        # Pooling 5
        pool5 = tf.nn.max_pool(
            h_conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool5"
        )

        if not include_top:
            return tf.contrib.layers.flatten(pool5)

        # Fully connected 6
        with tf.variable_scope("fc6", reuse=reuse):
            h_fc6, w_fc6, b_fc6 = vgg_fc(pool5, [7, 7, 512, 4096], tf.nn.relu)
        # Fully connected 7
        with tf.variable_scope("fc7", reuse=reuse):
            h_fc7, w_fc7, b_fc7 = vgg_fc(h_fc6, [1, 1, 4096, 4096], tf.nn.relu)
        # Fully connected 8
        with tf.variable_scope("fc8", reuse=reuse):
            h_fc8, w_fc8, b_fc8 = vgg_fc(h_fc7, [1, 1, 4096, 1000])
        return h_fc8
