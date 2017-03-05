# -*- coding: utf-8 -*-

import os
import unittest

import tensorflow as tf
from tensorflow.contrib.slim import nets
import numpy as np
from scipy.misc import imread, imresize
from keras.applications.vgg16 import VGG16

from tfmodel import vgg16


class TestVgg16(unittest.TestCase):

    def test_vgg16_with_keras(self):
        # Load sample image
        img_path = "img/tensorflow_logo.png"
        img = np.array([imresize(imread(img_path, mode="RGB"), [224, 224])], dtype=np.float32)
        # Try VGG 16 model converted for TensorFlow
        with tf.Graph().as_default() as g:
            model_tf = vgg16.Vgg16()
            img_ph = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
            model_tf.build_graph(img_ph)
            with tf.Session() as sess:
                model_tf.restore_variables(sess)
                p_tf = sess.run(tf.nn.softmax(model_tf.logits), feed_dict={img_ph: img})[0]
        # Try VGG 16 model included in Keras
        model = VGG16(weights='imagenet', include_top=True)
        print img.shape
        p_keras = model.predict(img[:, :, :, ::-1])[0]
        np.testing.assert_array_almost_equal(p_tf.flatten(), p_keras.flatten())

    def test_vgg16_with_tfslim(self):
        # Load sample image
        img_path = "img/tensorflow_logo.png"
        img = np.array([imresize(imread(img_path, mode="RGB"), [224, 224])], dtype=np.float32)
        # Try VGG 16 model converted for TensorFlow
        with tf.Graph().as_default() as g:
            model_tf = vgg16.Vgg16()
            img_ph = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
            model_tf.build_graph(img_ph)
            tf.summary.FileWriter(logdir="summary/tfmodel", graph=g)
            with tf.Session() as sess:
                model_tf.restore_variables(sess)
                logits_tf = sess.run(model_tf.logits, feed_dict={img_ph: img})[0]
        # Try VGG 16 model included in TF-Slim
        with tf.Graph().as_default() as g:
            x_ph = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
            net, end_points = nets.vgg.vgg_16(inputs=x_ph, num_classes=1000, is_training=False)
            saver = tf.train.Saver()
            tf.summary.FileWriter(logdir="summary/slim", graph=g)
            with tf.Session() as sess:
                saver.restore(sess, "{}/.tfmodel/vgg16/vgg_16.ckpt".format(os.environ["HOME"]))
                logits_slim = sess.run(net, feed_dict={x_ph: img})[0]
        np.testing.assert_array_almost_equal(logits_tf.flatten(), logits_slim.flatten())
