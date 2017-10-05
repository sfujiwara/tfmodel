# -*- coding: utf-8 -*-

import argparse
import os
import tensorflow as tf
import tfmodel
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--train_csv", type=str)
parser.add_argument("--test_csv", type=str)
parser.add_argument("--output_path", type=str, default="outputs")
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--n_classes", type=int, default=2)
parser.add_argument("--n_epochs", type=int, default=1)
args, unknown_args = parser.parse_known_args()

N_CLASSES = args.n_classes
BATCH_SIZE = args.batch_size
TRAIN_CSV = args.train_csv
TEST_CSV = args.test_csv
LEARNING_RATE = args.learning_rate
OUTPUT_PATH = args.output_path
N_EPOCHS = args.n_epochs

CHECKPOINT_DIR = os.path.join(OUTPUT_PATH, "checkpoints")


def build_queue(csv_file, num_epochs=None):
    with tf.name_scope("queue"):
        filename_queue = tf.train.string_input_producer([csv_file], num_epochs=num_epochs)
        reader = tf.TextLineReader(skip_header_lines=1)
        key, value = reader.read(filename_queue)
        img_file_path, label = tf.decode_csv(value, record_defaults=[[""], [1]])
        image = tf.image.decode_image(tf.read_file(img_file_path), channels=3)
        image = tf.image.resize_bicubic([image], [224, 224])[0]
        image.set_shape([224, 224, 3])
        image = tf.cast(image, tf.float32)
        # label = tf.to_int32(tf.string_to_number(label))
        label = tf.one_hot(label, depth=N_CLASSES)
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=BATCH_SIZE,
            num_threads=64,
            capacity=512,
            min_after_dequeue=0
        )
        return image_batch, label_batch


def get_input_fn(csv_file, n_epoch):

    def input_fn():
        image_batch, label_batch = build_queue(csv_file=csv_file, num_epochs=n_epoch)
        return {"images": image_batch}, label_batch

    return input_fn


def generate_csv(filenames, output, labels):
    image_file_paths = []
    image_labels = []
    for i, f in enumerate(filenames):
        files = tf.gfile.Glob(filename=f)
        l = [labels[i]] * len(files)
        image_file_paths.extend(files)
        image_labels.extend(l)
    result = zip(image_file_paths, image_labels)
    with tf.gfile.Open(output, mode="w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(result)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    run_config = tf.estimator.RunConfig().replace(
        save_summary_steps=1,
    )
    clf = tfmodel.estimator.VGG16Classifier(
        fc_units=[128],
        n_classes=2,
        model_dir="model",
        config=run_config
    )
    input_fn = get_input_fn(csv_file="img/train.csv", n_epoch=5)
    clf.train(input_fn=input_fn)
