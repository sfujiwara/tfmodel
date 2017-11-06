import tensorflow as tf


def ocsvm_model_fn(features, labels, mode, params, config=None):
    kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
        input_dim=params["rffm_input_dim"],
        output_dim=params["rffm_output_dim"],
        stddev=params["rffm_stddev"],
        name="rffm"
    )
    with tf.name_scope("feature_mapping"):
        mapped_features = kernel_mapper.map(features["x"])

    weight = tf.Variable(
        tf.truncated_normal([params["rffm_output_dim"], 1]),
        name="weight", dtype=tf.float32, trainable=True,
    )
    rho = tf.Variable(0, name="rho", dtype=tf.float32, trainable=True)
    tf.summary.scalar(name="rho", tensor=rho)
    tf.summary.histogram(name="weight", values=weight)

    y = tf.matmul(mapped_features, weight)
    decision_value = y - rho

    with tf.name_scope("regularizer"):
        regularizer = tf.nn.l2_loss(weight)
    with tf.name_scope("hinge_loss"):
        hinge_loss = tf.reduce_mean(tf.maximum(0., tf.add(rho, -y)))
    loss = hinge_loss + regularizer - tf.multiply(rho, params["nu"])

    tf.summary.scalar(name="regularizer", tensor=regularizer)
    tf.summary.histogram(name="decision_values", values=y)
    tf.summary.scalar(name="hinge_loss", tensor=hinge_loss)

    train_op = params["optimizer"].minimize(
        loss=loss,
        global_step=tf.train.get_global_step()
    )

    # Set eval metric
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=tf.sign(decision_value)),
            "recall": tf.metrics.recall(labels=labels, predictions=tf.sign(decision_value)),
            "precision": tf.metrics.precision(labels=labels, predictions=tf.sign(decision_value))
        }
    else:
        eval_metric_ops = None
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=decision_value,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        # scaffold=scaffold,
        # export_outputs={"prediction": export_outputs},
    )
    return estimator_spec


class OneClassSVM(tf.estimator.Estimator):

    def __init__(
        self,
        nu,
        rffm_input_dim,
        rffm_output_dim,
        rffm_stddev,
        optimizer=tf.train.ProximalAdagradOptimizer(1e-2),
        model_dir=None,
        config=None
    ):
        params = {
            "nu": nu,
            "rffm_input_dim": rffm_input_dim,
            "rffm_output_dim": rffm_output_dim,
            "rffm_stddev": rffm_stddev,
            "optimizer": optimizer,
        }
        super(OneClassSVM, self).__init__(
            model_fn=ocsvm_model_fn,
            model_dir=model_dir,
            params=params,
            config=config
        )


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import json
    import os

    tf_conf = {
      "cluster": {"master": ["localhost:2222"]},
      "task": {"index": 0, "type": "master"}
    }
    os.environ["TF_CONFIG"] = json.dumps(tf_conf)

    tf.logging.set_verbosity(tf.logging.DEBUG)
    x_train = np.random.multivariate_normal(mean=[1., 1.], cov=np.eye(2), size=100).astype(np.float32)
    x_eval = np.vstack([
        np.random.multivariate_normal(mean=[1., 1.], cov=np.eye(2), size=950).astype(np.float32),
        np.random.multivariate_normal(mean=[10., 10.], cov=np.eye(2), size=50).astype(np.float32)
    ])
    y_eval = np.array([1.]*950 + [-1.]*50).astype(np.float32)

    # feature_columns = [tf.feature_column.numeric_column("x", shape=[2])]

    config = tf.estimator.RunConfig(
        save_summary_steps=100,
        save_checkpoints_steps=2000,
        model_dir="outputs"
    )

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_train},
        y=None,
        shuffle=True,
        batch_size=32,
        num_epochs=None
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_eval},
        y=y_eval,
        shuffle=False,
        batch_size=32,
        num_epochs=1
    )

    clf = OneClassSVM(
        nu=0.1,
        rffm_input_dim=2,
        rffm_output_dim=2000,
        rffm_stddev=10.,
        optimizer=tf.train.ProximalAdagradOptimizer(1e-1),
        config=config
    )

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn, start_delay_secs=0, throttle_secs=1, exporters=None
    )
    tf.estimator.train_and_evaluate(estimator=clf, train_spec=train_spec, eval_spec=eval_spec)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_eval},
        y=y_eval,
        shuffle=False,
        num_epochs=1
    )
    result = np.array(list(clf.predict(predict_input_fn))).flatten()
    print(result)

    threshold = 0.
    ind_normal = result > threshold
    ind_outlier = result < threshold
    plt.plot(x_eval[ind_normal, 0], x_eval[ind_normal, 1], "x", label="Predicted as normal")
    plt.plot(x_eval[ind_outlier, 0], x_eval[ind_outlier, 1], "x", label="Predicted as outlier")
    plt.legend()
    plt.grid()
    plt.show()
