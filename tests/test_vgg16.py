# -*- coding: utf-8 -*-

import unittest

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from keras.applications.vgg16 import VGG16

from tfmodel import vgg16


class TestVgg16(unittest.TestCase):

    def test_vgg16(self):
        # Load sample image
        img_path = "img/tensorflow_logo.png"
        img = np.array([imresize(imread(img_path, mode="RGB"), [224, 224])], dtype=np.float32)
        # Try VGG 16 model converted for TensorFlow
        with tf.Graph().as_default() as g:
            model_tf = vgg16.Vgg16()
            img_ph = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
            prob = model_tf.inference(img_ph)
            saver = tf.train.Saver()
            with tf.Session() as sess:
                model_tf.restore_variables(sess, saver)
                p_tf = sess.run(prob, feed_dict={img_ph: img})[0]
        # Try VGG 16 model included in Keras
        model = VGG16(weights='imagenet', include_top=True)
        p_keras = model.predict(img)[0]
        np.testing.assert_array_almost_equal(p_tf, p_keras)
