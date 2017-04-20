# VGG 16

See [here](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) for details.

## Basic Usage

```python
import numpy as np
import tensorflow as tf
import tfmodel

img = np.random.normal(size=[1, 224, 224, 3])

with tf.Graph().as_default() as g:
    img_ph = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
    preprocessed_img = tfmodel.vgg.preprocess(img_ph)
    model_tf = tfmodel.vgg.Vgg16(preprocessed_img)
    with tf.Session() as sess:
        model_tf.restore_pretrained_variables(sess)
        p_tf = sess.run(tf.nn.softmax(model_tf.logits), feed_dict={img_ph: img})

print(p_tf)
```

## License

The pre-trained model is released under [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/).