# VGG 16

See [here](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) for details of VGG 16.

## Basic Usage

```python
import tfmodel

def train_input_fn():
    # Implement input pipeline for training data
    # x must be a dict or a Tensor with shape [batch_size, height, width, 3]
    # y must be a one-hot Tensor with shape [batch_size, n_classes]
    return {"images": xs}, ys

clf = tfmodel.estimator.VGG16Classifier(
    fc_units=[],
    n_classes=2,
    model_dir="outputs",
    pretrained_checkpoint_dir="models"
)

clf.train(input_fn=train_input_fn, steps=10000)
```

If `pretrained_checkpoint_dir` is specified, pre-trained checkpoint will be automatically downloaded from [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).

## License

The pre-trained model is released under [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/).