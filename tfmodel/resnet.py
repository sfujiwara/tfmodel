import tensorflow as tf


RESNET50_GRAPH_KEY = 'resnet50'


def resnet_conv2d(inputs, filters, kernel_size, strides, trainable):
    n_kernels = inputs.get_shape()[3].value
    w = tf.Variable(
        name='weights',
        initial_value=tf.random.normal([kernel_size, kernel_size, n_kernels, filters]),
        # shape=[kernel_size, kernel_size, n_kernels, filters],
        dtype=tf.float32,
        # initializer=tf.random_normal_initializer(stddev=0.01),
        trainable=trainable,
        collections=[RESNET50_GRAPH_KEY],
    )
    b = tf.Variable(
        name='biases',
        initial_value=tf.zeros(shape=[filters], dtype=tf.float32),
        # shape=[filters],
        dtype=tf.float32,
        # initializer=tf.zeros_initializer(),
        trainable=trainable,
        collections=[RESNET50_GRAPH_KEY],
    )
    h = tf.nn.conv2d(inputs, w, strides=[1, strides, strides, 1], padding='SAME')
    h = tf.nn.bias_add(h, b)
    h = tf.nn.relu(h)
    return h


def resnet_block(inputs, trainable):
    # type: (tf.Tensor, bool) -> tf.Tensor

    h = resnet_conv2d(inputs, filters=64, kernel_size=1, strides=1, trainable=trainable)
    h = resnet_conv2d(h, filters=64, kernel_size=3, strides=1, trainable=trainable)
    h = resnet_conv2d(h, filters=256, kernel_size=1, strides=1, trainable=trainable)
    return h


def resnet50_feature(inputs, trainable=True):

    # First convolution layer
    h = resnet_conv2d(inputs, filters=64, kernel_size=7, strides=2, trainable=trainable)
    # First pooling layer
    h = tf.nn.max_pool(h, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    h = resnet_block(h, trainable=True)
    return h


if __name__ == '__main__':
    x = tf.placeholder(dtype=tf.float32, shape=[32, 32, 32, 3])
    result = resnet50_feature(x)
    print(result)
    # import tensorflow.contrib.slim.nets as nets
    # result, _ = nets.resnet_v1.resnet_v1_50(x)
    # print(result)
    tf.summary.FileWriter('outputs', graph=tf.get_default_graph())
