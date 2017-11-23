# -*- coding: utf-8 -*-

import hashlib
import os
import tarfile
from six.moves import urllib
import tempfile
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import vgg


def download_vgg16_checkpoint(
        dest_directory=os.path.join(os.environ.get("HOME"), ".tfmodel", "models")
):
    data_url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
    dest_file = os.path.join(dest_directory, "vgg_16.ckpt")
    if tf.gfile.Exists(dest_file) and _verify_vgg16_checkpoint_hash(dest_file):
        tf.logging.info("{} already exists".format(dest_file))
    else:
        tarfile_path = os.path.join(tempfile.tempdir, os.path.basename(data_url))
        if tf.gfile.Exists(tarfile_path) and _verify_vgg16_tar_hash(tarfile_path):

            tf.logging.info("{} already exists".format(tarfile_path))
        else:
            tf.logging.info("downloading vgg16 checkpoint from {}".format(data_url))
            urllib.request.urlretrieve(data_url, filename=tarfile_path)
        tf.logging.info("extracting {}".format(tarfile_path))
        x = tarfile.open(name=tarfile_path, mode="r")
        fileobj = x.extractfile(x.getmembers()[0].name)
        tf.logging.info("saving vgg16 checkpoint to {}".format(dest_file))
        tf.gfile.MakeDirs(dest_directory)
        with tf.gfile.Open(dest_file, "w") as f:
            f.write(fileobj.read())


def _verify_vgg16_checkpoint_hash(checkpoint_path):
    with tf.gfile.Open(checkpoint_path) as f:
        is_valid = hashlib.md5(f.read()).hexdigest() == "c69996ee68fbd93d810407da7b3c0242"
    return is_valid


def _verify_vgg16_tar_hash(tar_path):
    with tf.gfile.Open(tar_path) as f:
        is_valid = hashlib.md5(f.read()).hexdigest() == "520bc6e4c73a89b5c0d8b9c4eaa8861f"
    return is_valid


def _default_resize_image_fn(img):
    img = tf.image.resize_bicubic([img], [224, 224])[0]
    # img = tf.image.resize_bicubic([img], [170, 225])[0]
    # img = tf.image.resize_image_with_crop_or_pad(img, 224, 224)
    img.set_shape([224, 224, 3])
    img = tf.cast(img, dtype=tf.uint8)
    img = tf.cast(img, dtype=tf.float32)
    return img


def embed(input_exps, output_dir, resize_image_fn=_default_resize_image_fn):
    metadata = [["file", "label"]]
    images = []
    with tf.Graph().as_default() as g:
        for i, exp in enumerate(input_exps):
            print i, exp
            file_list = tf.gfile.Glob(exp)
            print file_list
            for f in file_list:
                print f
                img = tf.image.decode_jpeg(tf.read_file(f), channels=3)
                img = resize_image_fn(img)
                metadata.append([f, str(i)])
                images.append(img)
        img_tensor = tf.stack(images)
        features = vgg.build_vgg16_graph(img_tensor=img_tensor, include_top=False)
        init_op = tf.global_variables_initializer()
        vgg16_saver = tf.train.Saver(tf.get_collection(vgg.VGG16_GRAPH_KEY))
        with tf.Session() as sess:
            sess.run(init_op)
            save_dir = os.path.join(os.environ.get("HOME", ""), ".tfmodel", "vgg16")
            download_vgg16_checkpoint(save_dir, vgg.MODEL_URL)
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
