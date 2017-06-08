# -*- coding: utf-8 -*-

import argparse
import os
import tensorflow as tf
import tfmodel

parser = argparse.ArgumentParser()
parser.add_argument("--train_csv", type=str)
parser.add_argument("--test_csv", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_classes", type=int, default=24)
parser.add_argument("--n_epochs", type=int, default=1)
args, unknown_args = parser.parse_known_args()

N_CLASSES = args.n_classes
BATCH_SIZE = args.batch_size
TRAIN_CSV = args.train_csv
TEST_CSV = args.test_csv
LEARNING_RATE = args.learning_rate
OUTPUT_PATH = args.output_path
N_EPOCHS = args.n_epochs


def build_queue(csv_file):
    with tf.name_scope("queue"):
        filename_queue = tf.train.string_input_producer([csv_file], num_epochs=N_EPOCHS)
        reader = tf.TextLineReader(skip_header_lines=1)
        key, value = reader.read(filename_queue)
        img_file_path, label = tf.decode_csv(value, record_defaults=[[""], [""]])
        image = tf.image.decode_image(tf.read_file(img_file_path), channels=3)
        image = tf.image.resize_bicubic([image], [224, 224])[0]
        image.set_shape([224, 224, 3])
        image = tf.cast(image, tf.float32)
        label = tf.to_int32(tf.string_to_number(label))
        label = tf.one_hot(label, depth=N_CLASSES)
        train_image_batch, train_label_batch = tf.train.batch(
            [image, label],
            batch_size=BATCH_SIZE,
            num_threads=128,
            capacity=512,
        )
        return train_image_batch, train_label_batch

# Build graph
with tf.Graph().as_default() as g:
    # Queue
    train_image_batch, train_label_batch = build_queue(TRAIN_CSV)
    # Build graph for forward step
    img_ph = tf.placeholder_with_default(train_image_batch, shape=[None, 224, 224, 3])
    label_ph = tf.placeholder_with_default(train_label_batch, shape=[None, N_CLASSES])
    nets = tfmodel.vgg.Vgg16(img_tensor=tfmodel.vgg.preprocess(img_ph), trainable=False, include_top=False)
    features = tf.reshape(nets.pool5, [-1, 7*7*512])
    logits = tf.layers.dense(features, N_CLASSES)
    outputs = tf.nn.softmax(logits)
    # Build loss graph
    with tf.name_scope("loss"):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=label_ph, logits=logits)
        tf.summary.scalar(tensor=loss, name="cross_entropy")
    # Build optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    # Initialization operation
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # Create summary writer
    train_writer = tf.summary.FileWriter(os.path.join(OUTPUT_PATH, "summaries", "train"), graph=g)
    summary_op = tf.summary.merge_all()

with tf.Session(graph=g) as sess:
    # Initialize all variables
    sess.run(init_op)
    # Load pre-trained VGG16
    nets.restore_pretrained_variables(sess)
    # Start populating the filename queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # Start training iteration
    try:
        for i in range(1000000):
            _, summary, l = sess.run([train_op, summary_op, loss])
            train_writer.add_summary(summary, i)
            tf.logging.info("Iteration: {0} Training Loss: {1}".format(i, l))
    except tf.errors.OutOfRangeError:
        tf.logging.info("Done training -- epoch limit reached")
    finally:
        coord.request_stop()
    coord.join(threads)
