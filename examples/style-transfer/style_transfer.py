# -*- coding: utf-8 -*-

import argparse
import os
import sys
import numpy as np
from scipy.misc import imread, imresize, imsave
import tensorflow as tf
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import tfmodel

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--content", type=str, default="img/contents/tensorflow_logo.jpg")
parser.add_argument("--style", type=str, default="img/styles/chouju_sumou.jpg")
parser.add_argument("--output_dir", type=str, default="outputs")
parser.add_argument("--content_weight", type=float, default=0.02)
parser.add_argument("--style_weight", type=float, default=0.98)
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

CONTENT_LAYERS = [
    # "vgg_16/conv4/conv4_2/Relu:0",
    "vgg_16/conv5/conv5_2/Relu:0",
]
STYLE_LAYERS = [
    "vgg_16/conv1/conv1_1/Relu:0",
    "vgg_16/conv2/conv2_1/Relu:0",
    "vgg_16/conv3/conv3_1/Relu:0",
    "vgg_16/conv4/conv4_1/Relu:0",
    "vgg_16/conv5/conv5_1/Relu:0",
]
PRE_TRAINED_MODEL_PATH = os.path.join(os.environ.get("HOME"), ".tfmodel", "vgg16", "vgg_16.ckpt")


def compute_target_style(style_img):
    with tf.Graph().as_default() as g1:
        width, height, _ = style_img.shape
        img_ph = tf.placeholder(tf.float32, [1, width, height, 3])
        tfmodel.vgg.build_vgg16_graph(img_tensor=tfmodel.vgg.preprocess(img_ph), include_top=False, trainable=False)
        style_layer_tensors = [g1.get_tensor_by_name(i) for i in STYLE_LAYERS]
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, PRE_TRAINED_MODEL_PATH)
            style_layers = sess.run(style_layer_tensors, feed_dict={img_ph: [style_img]})
    return style_layers


def compute_target_content(content_img):
    with tf.Graph().as_default() as g1:
        img_ph = tf.placeholder(tf.float32, [1, 224, 224, 3])
        _ = tfmodel.vgg.build_vgg16_graph(img_tensor=tfmodel.vgg.preprocess(img_ph), include_top=False, trainable=False)
        content_layer_tensors = [g1.get_tensor_by_name(i) for i in CONTENT_LAYERS]
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, PRE_TRAINED_MODEL_PATH)
            content_layers = sess.run(content_layer_tensors, feed_dict={img_ph: [content_img]})
    return content_layers


def build_content_loss(content_layer_tensors, target_content_layer_arrays):
    with tf.name_scope("content_loss"):
        content_losses = []
        for i in range(len(target_content_layer_arrays)):
            l = tf.losses.mean_squared_error(content_layer_tensors[i], target_content_layer_arrays[i])
            content_losses.append(l)
        content_loss = tf.reduce_sum(content_losses) * tf.constant(CONTENT_WEIGHT, name="content_weight")
        tf.summary.scalar("content_loss", content_loss)
    return content_loss


def build_style_loss(style_layer_tensors, target_style_layer_arrays):
    target_style_gram_arrays = []
    for sla in target_style_layer_arrays:
        f = np.reshape(sla, (-1, sla.shape[3]))
        target_style_gram_arrays.append(np.matmul(f.T, f) / sla.size)

    with tf.name_scope("style_loss"):
        style_loss = 0
        for i, slt in enumerate(style_layer_tensors):
            f = tf.reshape(slt, (-1, slt.get_shape()[3].value))
            style_gram = tf.matmul(tf.transpose(f), f) / slt.get_shape().num_elements()
            style_loss += tf.losses.mean_squared_error(style_gram, target_style_gram_arrays[i])
        style_loss *= tf.constant(STYLE_WEIGHT, name="style_weight")
        tf.summary.scalar("style_loss", style_loss)
    return style_loss


def build_total_variation_loss(img_tensor):
    with tf.name_scope("total_variation_loss"):
        h = img_tensor.get_shape()[1].value
        w = img_tensor.get_shape()[2].value
        tv_loss = tf.reduce_mean([
            tf.nn.l2_loss(img_tensor[:, 1:, :, :] - img_tensor[:, :w-1, :, :]),
            tf.nn.l2_loss(img_tensor[:, :, 1:, :] - img_tensor[:, :, :w-1, :])
        ]) * tf.constant(TV_WEIGHT, name="tv_weight")
        tf.summary.scalar("total_variation_loss", tv_loss)
    return tv_loss


tfmodel.util.maybe_download_and_extract(
    dest_directory=os.path.join(os.environ.get("HOME"), ".tfmodel", "vgg16"),
    data_url=tfmodel.vgg.MODEL_URL
)
content_img = imresize(imread(CONTENT, mode="RGB"), [224, 224]).astype(np.float32)
style_img = imread(STYLE, mode="RGB").astype(np.float32)
target_style_layer_arrays = compute_target_style(style_img)
target_content_layers = compute_target_content(content_img)

with tf.Graph().as_default() as g2:
    img_tensor = tf.Variable(tf.random_normal([1, 224, 224, 3], stddev=1., mean=128), name="generated_image")
    tf.summary.image("generated_image", img_tensor, max_outputs=100)
    tf.summary.image("content", np.expand_dims(content_img, axis=0))
    tf.summary.image("style", np.expand_dims(style_img, axis=0))
    tfmodel.vgg.build_vgg16_graph(img_tensor=tfmodel.vgg.preprocess(img_tensor), include_top=False, trainable=False)
    content_layer_tensors = [g2.get_tensor_by_name(i) for i in CONTENT_LAYERS]
    style_layer_tensors = [g2.get_tensor_by_name(i) for i in STYLE_LAYERS]

    # Build content loss
    content_loss = build_content_loss(
        content_layer_tensors=content_layer_tensors,
        target_content_layer_arrays=target_content_layers
    )
    # Build style loss
    style_loss = build_style_loss(
        style_layer_tensors=style_layer_tensors,
        target_style_layer_arrays=target_style_layer_arrays
    )
    # Build total variation loss
    tv_loss = build_total_variation_loss(img_tensor=img_tensor)
    # Build total loss
    with tf.name_scope("total_loss"):
        total_loss = content_loss + style_loss + tv_loss
        tf.summary.scalar("total_loss", total_loss)

    optim = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(total_loss)
    init_op = tf.global_variables_initializer()
    vgg16_saver = tf.train.Saver(tf.get_collection(tfmodel.vgg.VGG16_GRAPH_KEY))
    summary_writer = tf.summary.FileWriter("summary/neuralstyle", graph=g2)
    merged = tf.summary.merge_all()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(init_op)
        vgg16_saver.restore(sess, PRE_TRAINED_MODEL_PATH)
        for i in range(ITERATIONS):
            if i % SUMMARY_ITERATIONS == 0:
                if not os.path.exists(OUTPUT_DIR):
                    os.mkdir(OUTPUT_DIR)
                im = np.clip(sess.run(img_tensor)[0], 0, 255).astype(np.uint8)
                Image.fromarray(im).save(os.path.join(OUTPUT_DIR, "output-{}.jpg".format(i)), quality=95)
                # imsave(
                #     os.path.join(OUTPUT_DIR, "output-{}.jpg".format(i)),
                #     np.clip(sess.run(img_tensor)[0], 0, 255).astype(np.uint8)
                # )
                summary = sess.run(merged)
                summary_writer.add_summary(summary, i)
            _, t, c, s, tv = sess.run([optim, total_loss, content_loss, style_loss, tv_loss])
            print(
                "Iter: {0} TotalLoss: {1} ContentLoss: {2} StyleLoss: {3} TotalVariationLoss: {4}".format(
                    i, t, c, s, tv
                )
            )

