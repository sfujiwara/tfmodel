import os
import tensorflow as tf
import numpy as np
import tfmodel.util

MODEL_URL = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
VGG_MEAN = [123.68, 116.779, 103.939]
tf.GraphKeys.VGG16_VARIABLES = "vgg16"


def _preprocess(img_tensor):
    with tf.name_scope("preprocessing"):
        preprocessed_img_tensor = img_tensor - tf.constant(VGG_MEAN, name="vgg_mean")
    return preprocessed_img_tensor


def _vgg_conv2d(inputs, filters, trainable=True):
    n_kernels = inputs.get_shape()[3].value
    w = tf.get_variable(
        name="weights",
        shape=[3, 3, n_kernels, filters],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(),
        trainable=trainable,
        collections=[tf.GraphKeys.VGG16_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES],
    )
    b = tf.get_variable(
        name="biases",
        shape=[filters],
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=trainable,
        collections=[tf.GraphKeys.VGG16_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES],
    )
    h = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding="SAME"), b))
    return h, w, b


def build_vgg16_graph(img_tensor, reuse=False, trainable=True, include_top=True):
    img_tensor = _preprocess(img_tensor)
    with tf.variable_scope("vgg_16", reuse=reuse):
        # Convolution layers 1
        with tf.variable_scope("conv1"):
            with tf.variable_scope("conv1_1"):
                h_conv1_1, w_conv1_1, b_conv1_1 = _vgg_conv2d(img_tensor, 64, trainable)
            with tf.variable_scope("conv1_2"):
                h_conv1_2, w_conv1_2, b_conv1_2 = _vgg_conv2d(h_conv1_1, 64, trainable)
        # Pooling 1
        pool1 = tf.nn.max_pool(
            h_conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1"
        )
        # Convolution layers 2
        with tf.variable_scope("conv2"):
            with tf.variable_scope("conv2_1"):
                h_conv2_1, w_conv2_1, b_conv2_1 = _vgg_conv2d(pool1, 128, trainable)
            with tf.variable_scope("conv2_2"):
                h_conv2_2, w_conv2_2, b_conv2_2 = _vgg_conv2d(h_conv2_1, 128, trainable)
        # Pooling 2
        pool2 = tf.nn.max_pool(
            h_conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2"
        )
        # Convolution layers 3
        with tf.variable_scope("conv3"):
            with tf.variable_scope("conv3_1"):
                h_conv3_1, w_conv3_1, b_conv3_1 = _vgg_conv2d(pool2, 256, trainable)
            with tf.variable_scope("conv3_2"):
                h_conv3_2, w_conv3_2, b_conv3_2 = _vgg_conv2d(h_conv3_1, 256, trainable)
            with tf.variable_scope("conv3_3"):
                h_conv3_3, w_conv3_3, b_conv3_3 = _vgg_conv2d(h_conv3_2, 256, trainable)
        # Pooling 3
        pool3 = tf.nn.max_pool(
            h_conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool3"
        )
        # Convolution 4
        with tf.variable_scope("conv4"):
            with tf.variable_scope("conv4_1"):
                h_conv4_1, w_conv4_1, b_conv4_1 = _vgg_conv2d(pool3, 512, trainable)
            with tf.variable_scope("conv4_2"):
                h_conv4_2, w_conv4_2, b_conv4_2 = _vgg_conv2d(h_conv4_1, 512, trainable)
            with tf.variable_scope("conv4_3"):
                h_conv4_3, w_conv4_3, b_conv4_3 = _vgg_conv2d(h_conv4_2, 512, trainable)
        # Pooling 4
        pool4 = tf.nn.max_pool(
            h_conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool4"
        )
        # Convolution 5
        with tf.variable_scope("conv5"):
            with tf.variable_scope("conv5_1"):
                h_conv5_1, w_conv5_1, b_conv5_1 = _vgg_conv2d(pool4, 512, trainable)
            with tf.variable_scope("conv5_2"):
                h_conv5_2, w_conv5_2, b_conv5_2 = _vgg_conv2d(h_conv5_1, 512, trainable)
            with tf.variable_scope("conv5_3"):
                h_conv5_3, w_conv5_3, b_conv5_3 = _vgg_conv2d(h_conv5_2, 512, trainable)
        # Pooling 5
        pool5 = tf.nn.max_pool(
            h_conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool5"
        )


def vgg16_model_fn(features, labels, mode, params, config=None):
    build_vgg16_graph(features, trainable=False, reuse=False)
    pool5 = tf.get_default_graph().get_tensor_by_name("vgg_16/pool5:0")
    hidden = tf.contrib.layers.flatten(pool5)
    for n_unit in params["fc_units"]:
        hidden = tf.layers.dense(hidden, n_unit, activation=tf.nn.relu)
    logits = tf.layers.dense(hidden, params["n_classes"])
    prob = tf.nn.softmax(logits)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, label_smoothing=1e-7)
    optim = params["optimizer"]
    train_op = optim.minimize(loss)
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.VGG16_VARIABLES))

    def init_fn(scaffold, session):
        tf.logging.info("init_fn is successfully called")
        # Download weights
        save_dir = os.path.join(os.environ.get("HOME", ""), ".tfmodel", "vgg16")
        tfmodel.util.maybe_download_and_extract(dest_directory=save_dir, data_url=MODEL_URL)
        checkpoint_path = os.path.join(save_dir, "vgg_16.ckpt")
        saver.restore(session, checkpoint_path)

    scaffold = tf.train.Scaffold(init_fn=init_fn)
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=prob,
        loss=loss,
        train_op=train_op,
        scaffold=scaffold
    )
    return estimator_spec


def train_input_fn():
    img = np.random.normal(size=[32, 224, 224, 3])
    return tf.constant(img, dtype=tf.float32),  tf.one_hot(np.random.choice(2, 32), depth=2)


class VGG16Classifier(tf.estimator.Estimator):

    def __init__(
        self,
        fc_units,
        n_classes,
        optimizer=tf.train.ProximalAdagradOptimizer(1e-2),
        model_dir=None,
        config=None
    ):
        params = {
            "fc_units": fc_units,
            "n_classes": n_classes,
            "optimizer": optimizer,
        }
        super(VGG16Classifier, self).__init__(
            model_fn=vgg16_model_fn,
            model_dir=model_dir,
            params=params,
            config=config
        )


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    estimator_config = tf.contrib.learn.RunConfig(
        save_checkpoints_steps=100,
        save_summary_steps=50,
    )
    clf = VGG16Classifier(fc_units=[], n_classes=2, model_dir="outputs", config=estimator_config)
    validation_monitor = tf.contrib.learn.monitors.replace_monitors_with_hooks(
        [tf.contrib.learn.monitors.ValidationMonitor(
            input_fn=train_input_fn,
            eval_steps=1,
            every_n_steps=200,
            name=None
        )],
        clf
    )
    clf.train(input_fn=train_input_fn, steps=1, hooks=validation_monitor)
