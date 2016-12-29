# -*- coding: utf-8 -*-

import os
import subprocess

import tensorflow as tf
from tensorflow.contrib import layers

MODEL_URL1 = "https://github.com/sfujiwara/tfmodel/releases/download/v0.1/vgg16.data-00000-of-00001"
MODEL_URL2 = "https://github.com/sfujiwara/tfmodel/releases/download/v0.1/vgg16.index"
VGG_MEAN = [123.68, 116.779, 103.939]


class Vgg16:

    def __init__(self):
        pass

    def build_graph(self, img_tensor):
        # with tf.name_scope('preprocess') as scope:
        #     mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        #     img_ph = img_ph - mean
        self.conv1_1 = layers.convolution2d(inputs=img_tensor, num_outputs=64, kernel_size=3, padding="SAME", scope="conv1_1")
        self.conv1_2 = layers.convolution2d(inputs=self.conv1_1, num_outputs=64, kernel_size=3, padding="SAME", scope="conv1_2")
        self.pool1 = layers.max_pool2d(self.conv1_2, kernel_size=[2, 2], stride=[2, 2], padding="SAME", scope="pool1")
        self.conv2_1 = layers.convolution2d(inputs=self.pool1, num_outputs=128, kernel_size=3, padding="SAME", scope="conv2_1")
        self.conv2_2 = layers.convolution2d(inputs=self.conv2_1, num_outputs=128, kernel_size=3, padding="SAME", scope="conv2_2")
        self.pool2 = layers.max_pool2d(self.conv2_2, kernel_size=[2, 2], stride=[2, 2], padding="SAME", scope="pool2")
        self.conv3_1 = layers.convolution2d(inputs=self.pool2, num_outputs=256, kernel_size=3, padding="SAME", scope="conv3_1")
        self.conv3_2 = layers.convolution2d(inputs=self.conv3_1, num_outputs=256, kernel_size=3, padding="SAME", scope="conv3_2")
        self.conv3_3 = layers.convolution2d(inputs=self.conv3_2, num_outputs=256, kernel_size=3, padding="SAME", scope="conv3_3")
        self.pool3 = layers.max_pool2d(self.conv3_3, kernel_size=[2, 2], stride=[2, 2], padding="SAME", scope="pool3")
        self.conv4_1 = layers.convolution2d(inputs=self.pool3, num_outputs=512, kernel_size=3, padding="SAME", scope="conv4_1")
        self.conv4_2 = layers.convolution2d(inputs=self.conv4_1, num_outputs=512, kernel_size=3, padding="SAME", scope="conv4_2")
        self.conv4_3 = layers.convolution2d(inputs=self.conv4_2, num_outputs=512, kernel_size=3, padding="SAME", scope="conv4_3")
        self.pool4 = layers.max_pool2d(self.conv4_3, kernel_size=[2, 2], stride=[2, 2], padding="SAME", scope="pool4")
        self.conv5_1 = layers.convolution2d(inputs=self.pool4, num_outputs=512, kernel_size=3, padding="SAME", scope="conv5_1")
        self.conv5_2 = layers.convolution2d(inputs=self.conv5_1, num_outputs=512, kernel_size=3, padding="SAME", scope="conv5_2")
        self.conv5_3 = layers.convolution2d(inputs=self.conv5_2, num_outputs=512, kernel_size=3, padding="SAME", scope="conv5_3")
        self.pool5 = layers.max_pool2d(self.conv5_3, kernel_size=[2, 2], stride=[2, 2], padding="SAME", scope="pool5")
        self.fc6 = layers.fully_connected(inputs=layers.flatten(self.pool5), num_outputs=4096, scope="fc6")
        self.fc7 = layers.fully_connected(inputs=self.fc6, num_outputs=4096, scope="fc7")
        self.fc8 = layers.fully_connected(inputs=self.fc7, num_outputs=1000, activation_fn=None, scope="fc8")
        with tf.name_scope("prob"):
            self.prob = tf.nn.softmax(self.fc8)

    @staticmethod
    def restore_variables(session, saver):
        # Download weights
        save_dir = os.path.join(os.environ["HOME"], ".tfmodel", "vgg16")
        subprocess.call(["wget", "-nc", MODEL_URL1, "-P", save_dir])
        subprocess.call(["wget", "-nc", MODEL_URL2, "-P", save_dir])
        # Restore variables
        saver.restore(session, "{}/.tfmodel/vgg16/vgg16".format(os.environ["HOME"]))
