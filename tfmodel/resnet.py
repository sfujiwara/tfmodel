import typing
import tensorflow as tf


RESNET50_GRAPH_KEY = 'resnet50'


def resnet_conv2d(inputs, filters, kernel_size, strides, trainable):
    n_kernels = inputs.get_shape()[3].value
    w = tf.get_variable(
        name='weights',
        shape=[kernel_size, kernel_size, n_kernels, filters],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.01),
        trainable=trainable,
        collections=[RESNET50_GRAPH_KEY],
    )
    b = tf.get_variable(
        name='biases',
        shape=[filters],
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=trainable,
        collections=[RESNET50_GRAPH_KEY],
    )
    h = tf.nn.conv2d(inputs, w, strides=[1, strides, strides, 1], padding='SAME')
    h = tf.nn.bias_add(h, b)
    h = tf.nn.relu(h)
    return h


def resnet_block(inputs, filters, trainable):
    # type: (tf.Tensor, typing.List, bool) -> tf.Tensor

    with tf.variable_scope("conv1"):
        h = resnet_conv2d(inputs, filters=filters[0], kernel_size=1, strides=1, trainable=trainable)
    with tf.variable_scope("conv2"):
        h = resnet_conv2d(h, filters=filters[1], kernel_size=3, strides=1, trainable=trainable)
    with tf.variable_scope("conv3"):
        h = resnet_conv2d(h, filters=filters[2], kernel_size=1, strides=1, trainable=trainable)
    return h


def resnet50_feature(inputs, trainable=True):

    with tf.variable_scope("resnet_50_v1"):
        # First convolution layer
        with tf.variable_scope("conv1"):
            h = resnet_conv2d(inputs, filters=64, kernel_size=7, strides=2, trainable=trainable)
        # First pooling layer
        h = tf.nn.max_pool(h, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        with tf.variable_scope("block1"):
            with tf.variable_scope("unit_1"):
                h = resnet_block(h, filters=[64, 64, 256], trainable=trainable)
            with tf.variable_scope("unit_2"):
                h = resnet_block(h, filters=[64, 64, 256], trainable=trainable)
            with tf.variable_scope("unit_3"):
                h = resnet_block(h, filters=[64, 64, 256], trainable=trainable)
        with tf.variable_scope("block2"):
            with tf.variable_scope("unit_1"):
                h = resnet_block(h, filters=[128, 128, 512], trainable=trainable)
            with tf.variable_scope("unit_2"):
                h = resnet_block(h, filters=[128, 128, 512], trainable=trainable)
            with tf.variable_scope("unit_3"):
                h = resnet_block(h, filters=[128, 128, 512], trainable=trainable)
            with tf.variable_scope("unit_4"):
                h = resnet_block(h, filters=[128, 128, 512], trainable=trainable)
        with tf.variable_scope("block3"):
            with tf.variable_scope("unit_1"):
                h = resnet_block(h, filters=[256, 256, 1024], trainable=trainable)
            with tf.variable_scope("unit_2"):
                h = resnet_block(h, filters=[256, 256, 1024], trainable=trainable)
            with tf.variable_scope("unit_3"):
                h = resnet_block(h, filters=[256, 256, 1024], trainable=trainable)
            with tf.variable_scope("unit_4"):
                h = resnet_block(h, filters=[256, 256, 1024], trainable=trainable)
            with tf.variable_scope("unit_5"):
                h = resnet_block(h, filters=[256, 256, 1024], trainable=trainable)
            with tf.variable_scope("unit_6"):
                h = resnet_block(h, filters=[256, 256, 1024], trainable=trainable)
        with tf.variable_scope("block4"):
            with tf.variable_scope("unit_1"):
                h = resnet_block(h, filters=[512, 512, 2048], trainable=trainable)
            with tf.variable_scope("unit_2"):
                h = resnet_block(h, filters=[512, 512, 2048], trainable=trainable)
            with tf.variable_scope("unit_3"):
                h = resnet_block(h, filters=[512, 512, 2048], trainable=trainable)
    return h


if __name__ == '__main__':
    x = tf.placeholder(dtype=tf.float32, shape=[32, 224, 224, 3])
    result = resnet50_feature(x)
    print(result)
    # import tensorflow.contrib.slim.nets as nets
    # result, _ = nets.resnet_v1.resnet_v1_50(x)
    # print(result)
    # import tensorflow_hub as hub
    # m = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_50/classification/1")
    # result = m(x)
    tf.summary.FileWriter('outputs', graph=tf.get_default_graph())
