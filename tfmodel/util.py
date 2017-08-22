# -*- coding: utf-8 -*-

import os
import tarfile
from six.moves import urllib
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import vgg


def maybe_download_and_extract(dest_directory, data_url):
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split("/")[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(data_url, filepath)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def embed(input_csv, output_dir):
    metadata = [["file", "label"]]
    images = []
    with tf.Graph().as_default() as g:
        for row in tf.gfile.GFile.read(tf.gfile.Open(input_csv)).strip().splitlines():
            file_path, label = row.split(",")
            img = tf.image.decode_jpeg(tf.read_file(file_path), channels=3)
            img = _resize_image(img)
            metadata.append([file_path, label])
            images.append(img)
        img_tensor = tf.stack(images)
        print img_tensor
        features = vgg.build_vgg16_graph(img_tensor=img_tensor, include_top=False)
        init_op = tf.global_variables_initializer()
        vgg16_saver = tf.train.Saver(tf.get_collection(vgg.VGG16_GRAPH_KEY))
        with tf.Session() as sess:
            sess.run(init_op)
            save_dir = os.path.join(os.environ.get("HOME", ""), ".tfmodel", "vgg16")
            maybe_download_and_extract(save_dir, vgg.MODEL_URL)
            vgg16_saver.restore(sess, os.path.join(save_dir, "vgg_16.ckpt"))
            features_array = sess.run(features)
            print features_array
    with tf.Graph().as_default() as g:
        img_var = tf.Variable(features_array, name="images")
        saver = tf.train.Saver(var_list=[img_var])
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            projector_config = projector.ProjectorConfig()
            embedding = projector_config.embeddings.add()
            embedding.tensor_name = img_var.name
            embedding.metadata_path = "metadata.tsv"
            summary_write = tf.summary.FileWriter(output_dir)
            projector.visualize_embeddings(summary_writer=summary_write, config=projector_config)
            saver.save(sess, os.path.join(output_dir, "embeddings.ckpt"))
    # Convert list to TSV
    metadata = "\n".join(["\t".join(i) for i in metadata])
    tf.gfile.Open(os.path.join(output_dir, "metadata.tsv"), "w").write(metadata)


def _resize_image(img):
    img = tf.image.resize_bicubic([img], [170, 225])[0]
    img = tf.image.resize_image_with_crop_or_pad(img, 224, 224)
    img.set_shape([224, 224, 3])
    img = tf.cast(img, dtype=tf.uint8)
    img = tf.cast(img, dtype=tf.float32)
    return img
