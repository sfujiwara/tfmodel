# -*- coding: utf-8 -*-

import os
import unittest

import tensorflow as tf
from tensorflow.contrib.slim import nets
import numpy as np
from keras.applications.vgg16 import VGG16

import tfmodel


class TestVgg16(unittest.TestCase):

    def test_vgg16_with_keras(self):
        # Load sample image
        img = np.random.normal(size=[1, 224, 224, 3])
        # Try VGG 16 model converted for TensorFlow
        with tf.Graph().as_default() as g:
            img_ph = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
            model_tf = tfmodel.vgg.Vgg16(img_ph)
            with tf.Session() as sess:
                model_tf.restore_pretrained_variables(sess)
                p_tf = sess.run(tf.nn.softmax(model_tf.logits), feed_dict={img_ph: img})[0]
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
            model_tf = tfmodel.vgg.Vgg16(img_ph)
            tf.summary.FileWriter(logdir="summary/tfmodel", graph=g)
            with tf.Session() as sess:
                model_tf.restore_pretrained_variables(sess)
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

    def test_vgg16_hook(self):
        # Load sample image
        img = np.random.normal(size=[1, 224, 224, 3])
        # Try VGG 16 model converted for TensorFlow
        with tf.Graph().as_default() as g:
            img_ph = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
            model_tf = tfmodel.vgg.Vgg16(img_ph)
            model_load_hook = model_tf.create_model_load_hook()
            with tf.train.MonitoredTrainingSession(hooks=[model_load_hook]) as mon_sess:
                logits_tf = mon_sess.run(model_tf.logits, feed_dict={img_ph: img})[0]
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

    def test_vgg16_init_fn(self):
        # Load sample image
        img = np.random.normal(size=[1, 224, 224, 3])
        # Try VGG 16 model converted for TensorFlow
        with tf.Graph().as_default() as g:
            img_ph = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
            model_tf = tfmodel.vgg.Vgg16(img_ph)
            init_fn = model_tf.create_init_fn()
            scaffold = tf.train.Scaffold(init_fn=init_fn)
            with tf.train.MonitoredTrainingSession(scaffold=scaffold) as mon_sess:
                logits_tf = mon_sess.run(model_tf.logits, feed_dict={img_ph: img})[0]
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
