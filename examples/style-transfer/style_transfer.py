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
parser.add_argument("--content", type=str, default="img/tensorflow_logo.png")
parser.add_argument("--style", type=str, default="img/chouju_sumou.jpg")
parser.add_argument("--output_dir", type=str, default="outputs")
parser.add_argument("--content_weight", type=float, default=5e0*2)
parser.add_argument("--style_weight", type=float, default=5e2)
parser.add_argument("--iterations", type=int, default=3000)
parser.add_argument("--learning_rate", type=float, default=1e1)
args, unknown_args = parser.parse_known_args()

CONTENT = args.content
STYLE = args.style
OUTPUT_DIR = args.output_dir
CONTENT_WEIGHT = args.content_weight
STYLE_WEIGHT = args.style_weight
LEARNING_RATE = args.learning_rate
ITERATIONS = args.iterations

content_img = np.array([imresize(imread(CONTENT, mode="RGB"), [224, 224])], dtype=np.float32)
# content_img[content_img == 0.] = 254.
style_img = np.array([imresize(imread("img/chouju_sumou.jpg", mode="RGB"), [224, 224])], dtype=np.float32)

with tf.Graph().as_default() as g1:
    img_ph = tf.placeholder(tf.float32, [1, 224, 224, 3])
    net = tfmodel.vgg.Vgg16(img_tensor=img_ph)
    # content_layer_tensors = [net.h_conv4_2, net.h_conv5_2]
    content_layer_tensors = [net.h_conv4_2]
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
    net = tfmodel.vgg.Vgg16(img_tensor=img_tensor, trainable=False)
    # content_layer_tensors = [net.h_conv4_2, net.h_conv5_2]
    content_layer_tensors = [net.h_conv4_2]
    style_layer_tensors = [net.h_conv1_1, net.h_conv2_1, net.h_conv3_1, net.h_conv4_1, net.h_conv5_1]

    # Define content loss
    with tf.name_scope("content_loss"):
        content_losses = []
        for i in range(len(content_layers)):
            content_losses.append(tf.reduce_mean(tf.squared_difference(content_layer_tensors[i], content_layers[i])))
        content_loss = tf.reduce_sum(content_losses) * tf.constant(CONTENT_WEIGHT, name="content_weight")
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
        style_loss = tf.reduce_sum(style_losses) * tf.constant(STYLE_WEIGHT, name="style_weight")
        tf.summary.scalar("style_loss", style_loss)

    with tf.name_scope("total_loss"):
        total_loss = content_loss + style_loss
        tf.summary.scalar("total_loss", total_loss)
    optim = tf.train.AdamOptimizer(
        learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999, epsilon=1e-8
    ).minimize(total_loss)
    init_op = tf.global_variables_initializer()
    summary_writer = tf.summary.FileWriter("summary/neuralstyle", graph=g2)
    merged = tf.summary.merge_all()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(init_op)
        net.restore_pretrained_variables(session=sess)
        res = sess.run(content_layer_tensors)
        var = sess.run(img_tensor)
        for i in range(ITERATIONS):
            if i % 20 == 0:
                if not os.path.exists(OUTPUT_DIR):
                    os.mkdir(OUTPUT_DIR)
                imsave(os.path.join(OUTPUT_DIR, "output-{}.jpg".format(i)), sess.run(img_tensor)[0])
                summary = sess.run(merged)
                summary_writer.add_summary(summary, i)
            _, t, c, s = sess.run([optim, total_loss, content_loss, style_loss])
            print("total loss: {}".format(t))
            print("content loss: {}".format(c))
            print("style loss: {}".format(s))

