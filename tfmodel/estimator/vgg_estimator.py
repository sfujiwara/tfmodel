import os
import tensorflow as tf
import tfmodel


def vgg16_model_fn(features, labels, mode, params, config=None):
    if isinstance(features, dict):
        xs = features[list(features.keys())[0]]
    else:
        xs = features
    tfmodel.vgg.build_vgg16_graph(xs, trainable=False, reuse=False)
    pool5 = tf.get_default_graph().get_tensor_by_name("vgg_16/pool5:0")
    hidden = tf.layers.flatten(pool5)
    with tf.variable_scope("additional_layers"):
        for i, n_unit in enumerate(params["fc_units"]):
            hidden = tf.layers.dense(hidden, n_unit, activation=tf.nn.relu, name="fc{}".format(i))
        logits = tf.layers.dense(hidden, params["n_classes"], name="logits")
    prob = tf.nn.softmax(logits)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, label_smoothing=1e-7)
    optim = params["optimizer"]
    train_op = optim.minimize(loss=loss, global_step=tf.train.get_global_step())

    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=prob,
        loss=loss,
        train_op=train_op,
    )
    return estimator_spec


class VGG16Classifier(tf.estimator.Estimator):

    def __init__(
        self,
        fc_units,
        n_classes,
        optimizer=tf.train.ProximalAdagradOptimizer(1e-2),
        model_dir=None,
        config=None,
        warm_start_from=None,
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
            config=config,
            warm_start_from=warm_start_from,
        )


def vgg16_tpu_model_fn(features, labels, mode, params, config=None):
    if isinstance(features, dict):
        xs = features[list(features.keys())[0]]
    else:
        xs = features
    tfmodel.vgg.build_vgg16_graph(xs, trainable=False, reuse=False)
    pool5 = tf.get_default_graph().get_tensor_by_name("vgg_16/pool5:0")
    hidden = tf.layers.flatten(pool5)
    with tf.variable_scope("additional_layers"):
        for i, n_unit in enumerate(params["fc_units"]):
            hidden = tf.layers.dense(hidden, n_unit, activation=tf.nn.relu, name="fc{}".format(i))
        logits = tf.layers.dense(hidden, params["n_classes"], name="logits")
    prob = tf.nn.softmax(logits)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, label_smoothing=1e-7)
    optim = params["optimizer"]
    train_op = optim.minimize(loss=loss, global_step=tf.train.get_global_step())

    estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        predictions={"probabilities": prob},
        loss=loss,
        train_op=train_op,
    )
    return estimator_spec


class VGG16TPUClassifier(tf.contrib.tpu.TPUEstimator):

    def __init__(
        self,
        fc_units,
        n_classes,
        optimizer=tf.train.ProximalAdagradOptimizer(1e-2),
        model_dir=None,
        config=None,
        use_tpu=False,
        train_batch_size=32,
    ):
        if use_tpu:
            optimizer_ = tf.contrib.tpu.CrossShardOptimizer(optimizer)
        else:
            optimizer_ = optimizer
        params = {
            "fc_units": fc_units,
            "n_classes": n_classes,
            "optimizer": optimizer_,
        }

        super(VGG16TPUClassifier, self).__init__(
            model_fn=vgg16_model_fn,
            model_dir=model_dir,
            params=params,
            config=config,
            train_batch_size=train_batch_size,
            use_tpu=use_tpu
        )
