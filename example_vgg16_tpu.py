import numpy as np
import tensorflow as tf
import tfmodel


def train_input_fn():
    x = {"image": tf.constant(np.ones([32, 224, 224, 3]), dtype=np.float32, name="image")}
    y = tf.constant(np.zeros([32, 1000], dtype=np.float32), name="label")
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    return ds


def tpu_train_input_fn(params):
    x = {"image": tf.constant(np.ones([32, 224, 224, 3]), dtype=np.float32, name="image")}
    y = tf.constant(np.zeros([32, 1000], dtype=np.float32), name="label")
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.repeat().apply(
        tf.contrib.data.batch_and_drop_remainder(32)
    )
    return ds


def main():
    tpu_config = tf.contrib.tpu.TPUConfig(num_shards=8)
    estimator_config = tf.contrib.tpu.RunConfig(
        master="",
        save_checkpoints_steps=1000,
        save_summary_steps=1000,
        session_config=tf.ConfigProto(log_device_placement=True),
        model_dir="model",
        tpu_config=tpu_config,
    )
    clf = tfmodel.estimator.VGG16TPUClassifier(
        fc_units=[],
        n_classes=1000,
        optimizer=tf.train.GradientDescentOptimizer(1e-2),
        model_dir="model",
        config=estimator_config,
        train_batch_size=32
    )
    clf.train(input_fn=tpu_train_input_fn, max_steps=2)


if __name__ == "__main__":
    main()
