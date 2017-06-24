import tensorflow as tf


class ModelLoadHook(tf.train.SessionRunHook):

    def __init__(self, checkpoint, saver):
        self.saver = saver
        self.checkpoint = checkpoint

    def after_create_session(self, session, coord):
        self.saver.restore(session, self.checkpoint)
