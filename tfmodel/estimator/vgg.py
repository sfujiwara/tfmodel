import os
import tensorflow as tf
import numpy as np
import tfmodel


def vgg16_model_fn(features, labels, mode, params, config=None):
    tfmodel.vgg.build_vgg16_graph(features["images"], trainable=False, reuse=False)
    pool5 = tf.get_default_graph().get_tensor_by_name("vgg_16/pool5:0")
    hidden = tf.contrib.layers.flatten(pool5)
    with tf.variable_scope("additional_layers"):
        for n_unit in params["fc_units"]:
            hidden = tf.layers.dense(hidden, n_unit, activation=tf.nn.relu)
        logits = tf.layers.dense(hidden, params["n_classes"])
    prob = tf.nn.softmax(logits)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, label_smoothing=1e-7)
    optim = params["optimizer"]
    train_op = optim.minimize(loss)
    saver = tf.train.Saver(var_list=tf.get_collection(tfmodel.vgg.VGG16_GRAPH_KEY))

    def init_fn(scaffold, session):
        tf.logging.info("init_fn is successfully called")
        # Download weights
        save_dir = os.path.join(os.environ.get("HOME", ""), ".tfmodel", "vgg16")
        tfmodel.util.maybe_download_and_extract(dest_directory=save_dir, data_url=tfmodel.vgg.MODEL_URL)
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
    return tf.constant(img, dtype=tf.float32), tf.one_hot(np.random.choice(2, 32), depth=2)


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
