# -*- coding: utf-8 -*-

import tensorflow as tf
import tfmodel

N_CLASS = 2
BATCH_SIZE = 2

# Build graph
with tf.Graph().as_default() as g:
    # Queue
    with tf.name_scope("queue"):
        filename_queue = tf.train.string_input_producer(["data/train.csv"])
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)
        img_file_path, label = tf.decode_csv(value, record_defaults=[[""], [""]])
        image = tf.image.decode_image(tf.read_file(img_file_path), channels=3)
        image = tf.image.resize_bicubic([image], [224, 224])[0]
        image.set_shape([224, 224, 3])
        image = tf.cast(image, tf.float32)
        label = tf.to_int32(tf.string_to_number(label))
        label = tf.one_hot(label, depth=N_CLASS)
        train_image_batch, train_label_batch = tf.train.batch(
            [image, label],
            batch_size=BATCH_SIZE
        )
    # Build graph for forward step
    img_ph = tf.placeholder_with_default(train_image_batch, shape=[None, 224, 224, 3])
    label_ph = tf.placeholder_with_default(train_label_batch, shape=[None, N_CLASS])
    nets = tfmodel.vgg.Vgg16(img_tensor=img_ph, trainable=False, include_top=False)
    features = tf.reshape(nets.pool5, [-1, 7*7*512])
    logits = tf.layers.dense(features, N_CLASS)
    outputs = tf.nn.softmax(logits)
    # Build loss graph
    with tf.name_scope("cross_entropy"):
        loss = -tf.reduce_mean(tf.log(outputs) * label_ph)
        tf.summary.scalar(tensor=loss, name="loss")
    # Build optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)
    # Initialization operation
    init_op = tf.global_variables_initializer()
    # Create summary writer
    tf.summary.FileWriter(logdir="summary", graph=g)
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("summary/train", g)

with tf.Session(graph=g) as sess:
    # Initialize all variables
    sess.run(init_op)
    # Load pre-trained VGG16
    nets.restore_pretrained_variables(sess)
    # Start populating the filename queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(10):
        _, summary, l = sess.run([train_op, summary_op, loss])
        train_writer.add_summary(summary, i)
    coord.request_stop()
    coord.join(threads)
