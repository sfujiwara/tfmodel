# tfmodel

This module includes pre-trained models converted for TensorFlow.

## Requirements

tfmodel requires nothing but TensorFlow.
SciPy and Keras are required only for unit test.

## Models

### VGG 16

wget -nc https://github.com/sfujiwara/tfmodel/releases/download/v0.1/export.data-00000-of-00001 -P ~/.tfmodel/vgg16
wget -nc https://github.com/sfujiwara/tfmodel/releases/download/v0.1/export.index -P ~/.tfmodel/vgg16

## Unit Test

```
python -m unittest discover -v tests
```

## License

This module itself is released under MIT license.
Note that weights of existing pre-trained models follow each licenses.