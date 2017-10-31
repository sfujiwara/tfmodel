import tensorflow as tf


def ocsvm_model_fn(features, labels, mode, params, config=None):
    kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
        input_dim=params["rffm_input_dim"],
        output_dim=params["rffm_output_dim"],
        stddev=params["rffm_stddev"],
        name="rffm"
    )
    mapped_features = kernel_mapper.map(features["x"])
    weight = tf.get_variable(
        name="weight",
        shape=[params["rffm_output_dim"], 1],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(),
        trainable=True,
    )
    rho = tf.Variable(0, name="rho", dtype=tf.float32, trainable=True)
    tf.summary.scalar(name="rho", tensor=rho)
    tf.summary.histogram(name="weight", values=weight)

    y = tf.matmul(mapped_features, weight)
    with tf.name_scope("regularizer"):
        regularizer = tf.nn.l2_loss(weight)
    with tf.name_scope("hinge_loss"):
        hinge_loss = tf.reduce_mean(tf.maximum(0., tf.add(rho, -y)))
    loss = hinge_loss + regularizer - tf.multiply(rho, params["nu"])

    tf.summary.scalar(name="regularizer", tensor=regularizer)
    tf.summary.histogram(name="decisionValues", values=y)
    tf.summary.scalar(name="hinge_loss", tensor=hinge_loss)

    train_op = params["optimizer"].minimize(
        loss=loss,
        global_step=tf.train.get_global_step()
    )
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=y-rho,
        loss=loss,
        train_op=train_op,
        # eval_metric_ops=eval_metric_ops,
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

    tf.logging.set_verbosity(tf.logging.DEBUG)
    x_normal = np.random.multivariate_normal(mean=[1., 1.], cov=np.eye(2), size=95).astype(np.float32)
    x_outlier = np.random.multivariate_normal(mean=[10., 10.], cov=np.eye(2), size=5).astype(np.float32)
    x = np.vstack([x_normal, x_outlier])

    # feature_columns = [tf.feature_column.numeric_column("x", shape=[2])]

    config = tf.estimator.RunConfig(
        save_summary_steps=10,
        model_dir="output"
    )

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x},
        y=None,
        shuffle=True,
        batch_size=32,
        num_epochs=5000
    )

    clf = OneClassSVM(
        nu=0.3,
        rffm_input_dim=2,
        rffm_output_dim=2000,
        rffm_stddev=10.,
        optimizer=tf.train.ProximalAdagradOptimizer(1e-3),
        config=config
    )
    clf.train(input_fn=train_input_fn)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x},
        y=None,
        shuffle=False,
        num_epochs=1
    )
    result = np.array(list(clf.predict(predict_input_fn))).flatten()
    print(result)

    # plt.plot(x[:, 0], x[:, 1], "x")
    plt.plot(result)
    plt.show()
