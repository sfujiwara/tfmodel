# -*- coding: utf-8 -*-

import argparse
import os
import sys
import numpy as np
from scipy.misc import imread, imresize, imsave
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import tfmodel

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--content", type=str, default="img/contents/tensorflow_logo.jpg")
parser.add_argument("--style", type=str, default="img/styles/chouju_sumou.jpg")
parser.add_argument("--output_dir", type=str, default="outputs")
parser.add_argument("--content_weight", type=float, default=0.05)
parser.add_argument("--style_weight", type=float, default=0.95)
parser.add_argument("--tv_weight", type=float, default=0.0001)
parser.add_argument("--iterations", type=int, default=3000)
parser.add_argument("--learning_rate", type=float, default=1e1)
parser.add_argument("--summary_iterations", type=int, default=20)
args, unknown_args = parser.parse_known_args()

CONTENT = args.content
STYLE = args.style
OUTPUT_DIR = args.output_dir
CONTENT_WEIGHT = args.content_weight
STYLE_WEIGHT = args.style_weight
TV_WEIGHT = args.tv_weight
LEARNING_RATE = args.learning_rate
ITERATIONS = args.iterations
SUMMARY_ITERATIONS = args.summary_iterations

content_img = np.array([imresize(imread(CONTENT, mode="RGB"), [224, 224])], dtype=np.float32)
style_img = np.array([imresize(imread(STYLE, mode="RGB"), [224, 224])], dtype=np.float32)

# Compute target content and target style
with tf.Graph().as_default() as g1:
    img_ph = tf.placeholder(tf.float32, [1, 224, 224, 3])
    net = tfmodel.vgg.Vgg16(img_tensor=tfmodel.vgg.preprocess(img_ph))
    content_layer_tensors = [net.h_conv4_2, net.h_conv5_2]
    style_layer_tensors = [net.h_conv1_1, net.h_conv2_1, net.h_conv3_1, net.h_conv4_1, net.h_conv5_1]
    with tf.Session() as sess:
        net.restore_pretrained_variables(session=sess)
        content_layers = sess.run(content_layer_tensors, feed_dict={img_ph: content_img})
        style_layers = sess.run(style_layer_tensors, feed_dict={img_ph: style_img})

with tf.Graph().as_default() as g2:
    img_tensor = tf.Variable(tf.random_normal([1, 224, 224, 3], stddev=0.256))
    tf.summary.image("generated_image", img_tensor, max_outputs=100)
    tf.summary.image("content", content_img)
    tf.summary.image("style", style_img)
    net = tfmodel.vgg.Vgg16(img_tensor=tfmodel.vgg.preprocess(img_tensor), trainable=False)
    content_layer_tensors = [net.h_conv4_2, net.h_conv5_2]
    style_layer_tensors = [net.h_conv1_1, net.h_conv2_1, net.h_conv3_1, net.h_conv4_1, net.h_conv5_1]

    # Build content loss
    with tf.name_scope("content_loss"):
        content_losses = []
        for i in range(len(content_layers)):
            content_losses.append(tf.reduce_mean(tf.squared_difference(content_layer_tensors[i], content_layers[i])))
        content_loss = tf.reduce_sum(content_losses) * tf.constant(CONTENT_WEIGHT, name="content_weight")
        tf.summary.scalar("content_loss", content_loss)

    # Build style loss
    with tf.name_scope("style_loss"):
        style_losses = []
        for i in range(len(style_layers)):
            # Compute target gram matrix
            features = np.reshape(style_layers[i], (-1, style_layers[i].shape[3]))
            style_gram = np.matmul(features.T, features) / features.size
            # Build style tensor
            _, height, width, number = map(lambda x: x.value, style_layer_tensors[i].get_shape())
            size = height * width * number
            feats = tf.reshape(style_layer_tensors[i], (-1, number))
            gram = tf.matmul(tf.transpose(feats), feats) / size
            style_losses.append(tf.nn.l2_loss(gram - style_gram) / size)
        style_loss = tf.reduce_sum(style_losses) * tf.constant(STYLE_WEIGHT, name="style_weight")
        tf.summary.scalar("style_loss", style_loss)

    # Build total variation loss
    with tf.name_scope("total_variation_loss"):
        h = img_tensor.get_shape()[1].value
        w = img_tensor.get_shape()[2].value
        tv_loss = tf.reduce_mean([
            tf.nn.l2_loss(img_tensor[:, 1:, :, :] - img_tensor[:, :w-1, :, :]),
            tf.nn.l2_loss(img_tensor[:, :, 1:, :] - img_tensor[:, :, :w-1, :])
        ]) * tf.constant(TV_WEIGHT, name="tv_weight")
        tf.summary.scalar("total_variation_loss", tv_loss)

    # Build total loss
    with tf.name_scope("total_loss"):
        total_loss = content_loss + style_loss + tv_loss
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
        for i in range(ITERATIONS):
            if i % SUMMARY_ITERATIONS == 0:
                if not os.path.exists(OUTPUT_DIR):
                    os.mkdir(OUTPUT_DIR)
                imsave(os.path.join(OUTPUT_DIR, "output-{}.jpg".format(i)), sess.run(img_tensor)[0])
                summary = sess.run(merged)
                summary_writer.add_summary(summary, i)
            _, t, c, s, tv = sess.run([optim, total_loss, content_loss, style_loss, tv_loss])
            print(
                "Iter: {0} TotalLoss: {1} ContentLoss: {2} StyleLoss: {3} TotalVariationLoss: {4}".format(
                    i, t, c, s, tv
                )
            )

