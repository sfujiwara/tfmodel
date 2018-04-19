import os
import tensorflow as tf
import tfmodel


def metric_fn(labels, logits):
    n_classes = logits.shape[1].value
    n_classes = min(n_classes, 10)
    with tf.name_scope("metrics_accuracy"):
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=tf.argmax(labels, 1), predictions=tf.argmax(logits, 1)),
            "mean_par_accuracy": tf.metrics.mean_per_class_accuracy(
                labels=tf.argmax(labels, 1), predictions=tf.argmax(logits, 1), num_classes=n_classes
            ),
        }
    # Add recalls of each classes to eval metrics
    with tf.name_scope("metrics_recall"):
        for k in [1, 3]:
            for i in range(n_classes):
                eval_metric_ops["recall_at_{}/class_{}".format(k, i)] = tf.metrics.recall_at_k(
                    labels=tf.argmax(labels, 1), predictions=logits, k=k, class_id=i
                )
    # Add precisions of each classes to eval metrics
    with tf.name_scope("metrics_precision"):
        for k in [1]:
            for i in range(n_classes):
                eval_metric_ops["precision_at_{}/class_{}".format(k, i)] = tf.metrics.sparse_precision_at_k(
                    labels=tf.argmax(labels, 1), predictions=logits, k=k, class_id=i
                )
    return eval_metric_ops


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
        eval_metric_ops=metric_fn(labels=labels, logits=logits)
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
        metric_fn=(metric_fn, [labels, logits])
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
