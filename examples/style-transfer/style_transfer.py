# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from scipy.misc import imread, imresize, imsave
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import tfmodel

CONTENT_WEIGHT = 1.
STYLE_WEIGHT = 1.
LEARNING_RATE = 1.

content_img = np.array([imresize(imread("img/tensorflow_logo.png", mode="RGB"), [224, 224])], dtype=np.float32)
content_img[content_img == 0.] = 254.
style_img = np.array([imresize(imread("img/chouju_sumou.jpg", mode="RGB"), [224, 224])], dtype=np.float32)

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
    tf.summary.image("generated_image", img_tensor, max_outputs=100)
    tf.summary.image("content", content_img)
    tf.summary.image("style", style_img)
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
        tf.summary.scalar("content_loss", content_loss)
    # Define style loss
    with tf.name_scope("style_loss"):
        style_losses = []
        for i in range(len(style_layers)):
            # Compute target gram matrix
            features = np.reshape(style_layers[i], (-1, style_layers[i].shape[3]))
            style_gram = np.matmul(features.T, features) / features.size
            # Define style tensor
            _, height, width, number = map(lambda x: x.value, style_layer_tensors[i].get_shape())
            size = height * width * number
            feats = tf.reshape(style_layer_tensors[i], (-1, number))
            gram = tf.matmul(tf.transpose(feats), feats) / size
            style_losses.append(tf.nn.l2_loss(gram - style_gram) / size)
        style_loss = tf.reduce_sum(style_losses)
        tf.summary.scalar("style_loss", style_loss)
    with tf.name_scope("total_loss"):
        total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss
        tf.summary.scalar("total_loss", total_loss)
    optim = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(total_loss)
    init_op = tf.global_variables_initializer()
    summary_writer = tf.summary.FileWriter("summary/neuralstyle", graph=g2)
    merged = tf.summary.merge_all()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(init_op)
        net.restore_pretrained_variables(session=sess)
        res = sess.run(content_layer_tensors)
        var = sess.run(img_tensor)
        for i in range(3000):
            if i % 20 == 0:
                imsave("output-{}.jpg".format(i), sess.run(img_tensor)[0])
                summary = sess.run(merged)
                summary_writer.add_summary(summary, i)
            _, t, c, s = sess.run([optim, total_loss, content_loss, style_loss])
            print("total loss: {}".format(t))
            print("content loss: {}".format(c))
            print("style loss: {}".format(s))

