import os
import tensorflow as tf
import tfmodel


def vgg16_model_fn(features, labels, mode, params, config=None):
    if isinstance(features, dict):
        xs = features[features.keys()[0]]
    else:
        xs = features
    import IPython;IPython.embed()
    tfmodel.vgg.build_vgg16_graph(xs, trainable=False, reuse=False)
    pool5 = tf.get_default_graph().get_tensor_by_name("vgg_16/pool5:0")
    hidden = tf.contrib.layers.flatten(pool5)
    with tf.variable_scope("additional_layers"):
        for i, n_unit in enumerate(params["fc_units"]):
            hidden = tf.layers.dense(hidden, n_unit, activation=tf.nn.relu, name="fc{}".format(i))
        logits = tf.layers.dense(hidden, params["n_classes"], name="logits")
    prob = tf.nn.softmax(logits)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, label_smoothing=1e-7)
    optim = params["optimizer"]
    train_op = optim.minimize(loss=loss, global_step=tf.train.get_global_step())
    saver = tf.train.Saver(var_list=tf.get_collection(tfmodel.vgg.VGG16_GRAPH_KEY))

    def init_fn(scaffold, session):
        pretrained_checkpoint_dir = params["pretrained_checkpoint_dir"]
        if pretrained_checkpoint_dir is not None:
            tfmodel.util.download_vgg16_checkpoint(pretrained_checkpoint_dir)
            saver.restore(session, os.path.join(pretrained_checkpoint_dir, "vgg_16.ckpt"))

    scaffold = tf.train.Scaffold(init_fn=init_fn)
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=prob,
        loss=loss,
        train_op=train_op,
        scaffold=scaffold
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
        pretrained_checkpoint_dir=None
    ):
        params = {
            "fc_units": fc_units,
            "n_classes": n_classes,
            "optimizer": optimizer,
            "pretrained_checkpoint_dir": pretrained_checkpoint_dir
        }
        super(VGG16Classifier, self).__init__(
            model_fn=vgg16_model_fn,
            model_dir=model_dir,
            params=params,
            config=config
        )
