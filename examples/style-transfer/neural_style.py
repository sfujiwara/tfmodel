# -*- coding: utf-8 -*-

import numpy as np
from scipy.misc import imread, imresize, imsave
import tensorflow as tf
import tfmodel

content_img = np.array([imresize(imread("examples/img/tensorflow_logo.png", mode="RGB"), [224, 224])], dtype=np.float32)
style_img = np.array([imresize(imread("examples/img/chouju_sumou.jpg", mode="RGB"), [224, 224])], dtype=np.float32)

with tf.Graph().as_default() as g1:
    img_ph = tf.placeholder(tf.float32, [1, 224, 224, 3])
    net = tfmodel.vgg.Vgg16(img_tensor=img_ph)
    content_layer_tensors = [net.h_conv4_2, net.h_conv5_2]
    style_layer_tensors = [net.h_conv1_1, net.h_conv2_1, net.h_conv3_1, net.h_conv4_1, net.h_conv5_1]
    with tf.Session() as sess:
        net.restore_pretrained_variables(session=sess)
        content_layers = sess.run(content_layer_tensors, feed_dict={img_ph: content_img})
        style_layers = sess.run(style_layer_tensors, feed_dict={img_ph: style_img})

with tf.Graph().as_default() as g2:
    img_tensor = tf.Variable(tf.random_normal([1, 224, 224, 3]))
    # img_tensor = tf.Variable(content_img)
    net = tfmodel.vgg.Vgg16(img_tensor=img_tensor, trainable=False)
    content_layer_tensors = [net.h_conv4_2, net.h_conv5_2]
    style_layer_tensors = [net.h_conv1_1, net.h_conv2_1, net.h_conv3_1, net.h_conv4_1, net.h_conv5_1]
    # Define content loss
    with tf.name_scope("content_loss"):
        content_losses = []
        for i in range(len(content_layers)):
            content_losses.append(
                tf.reduce_mean(tf.squared_difference(content_layer_tensors[i], content_layers[i]))
            )
        content_loss = tf.reduce_sum(content_losses)
    # Define style loss
    with tf.name_scope("style_loss"):
        for i in range(len(style_layers)):
            # TODO: compute style loss
            pass
    optim = tf.train.AdamOptimizer(learning_rate=1e-1).minimize(content_loss)
    init_op = tf.global_variables_initializer()
    tf.summary.FileWriter("summary/neuralstyle", graph=g2)
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(init_op)
        net.restore_pretrained_variables(session=sess)
        res = sess.run(content_layer_tensors)
        var = sess.run(img_tensor)
        for i in range(1000):
            print sess.run(content_losses)
            if i % 10 == 0:
                imsave("output-{}.jpg".format(i), sess.run(img_tensor)[0])
            sess.run(optim)
