# -*- coding: utf-8 -*-

import os
import unittest

import tensorflow as tf
from tensorflow.contrib.slim import nets
import numpy as np
from keras.applications.vgg16 import VGG16

import tfmodel

TFMODEL_DIR = os.path.join(os.environ.get("HOME"), ".tfmodel")


class TestVgg16(unittest.TestCase):

    def setUp(self):
        tfmodel.util.maybe_download_and_extract(os.path.join(TFMODEL_DIR, "vgg16"), tfmodel.vgg.MODEL_URL)

    def test_vgg16_with_keras(self):
        # Load sample image
        img = np.random.normal(size=[1, 224, 224, 3])
        # Try VGG 16 model converted for TensorFlow
        with tf.Graph().as_default() as g:
            img_ph = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
            logits_tf = tfmodel.vgg.build_vgg16_graph(img_tensor=img_ph, include_top=True)
            saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess=sess, save_path=os.path.join(TFMODEL_DIR, "vgg16", "vgg_16.ckpt"))
                p_tf = sess.run(tf.nn.softmax(logits=logits_tf), feed_dict={img_ph: img})[0]
        # Try VGG 16 model included in Keras
        model = VGG16(weights="imagenet", include_top=True)
        p_keras = model.predict(img[:, :, :, ::-1])[0]
        np.testing.assert_array_almost_equal(p_tf.flatten(), p_keras.flatten())

    def test_vgg16_with_tfslim(self):
        # Load sample image
        img = np.random.normal(size=[1, 224, 224, 3])
        # Try VGG 16 model converted for TensorFlow
        with tf.Graph().as_default() as g:
            img_ph = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
            logits_tf = tfmodel.vgg.build_vgg16_graph(img_tensor=img_ph, include_top=True)
            saver = tf.train.Saver()
            tf.summary.FileWriter(logdir="summary/tfmodel", graph=g)
            with tf.Session() as sess:
                saver.restore(sess=sess, save_path=os.path.join(TFMODEL_DIR, "vgg16", "vgg_16.ckpt"))
                logits_tf = sess.run(logits_tf, feed_dict={img_ph: img})[0]
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
