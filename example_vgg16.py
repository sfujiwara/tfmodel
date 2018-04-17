import numpy as np
import tensorflow as tf
import tfmodel


def train_input_fn():
    x = {"image": tf.constant(np.ones([32, 224, 224, 3]), dtype=np.float32, name="image")}
    y = tf.constant(np.zeros([32, 1000], dtype=np.float32), name="label")
    return x, y


def main():
    warm_start_settings = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from="vgg_16.ckpt",
        vars_to_warm_start="vgg_16*",
    )
    clf = tfmodel.estimator.VGG16Classifier(
        fc_units=[],
        n_classes=1000,
        optimizer=tf.train.GradientDescentOptimizer(1e-2),
        model_dir="model",
        config=None,
        warm_start_from=warm_start_settings,
    )
    clf.train(input_fn=train_input_fn, max_steps=2)


if __name__ == "__main__":
    main()
