# One-Class Support Vector Machine

## Basic Usage

```python
import tensorflow as tf
import tfmodel


def train_input_fn():
    # Implement input pipeline for training data
    # x must be a dict to a Tensor
    # y is None since One-Class SVM is unsupervised learning 
    return {"x": xs}, None


feature_columns = [tf.feature_column.numeric_column("x", shape=[2])]

clf = tfmodel.estimator.OneClassSVM(
    feature_columns=feature_columns,
    nu=0.1,
    rffm_input_dim=2,
    rffm_output_dim=2000,
    rffm_stddev=10.,
    optimizer=tf.train.ProximalAdagradOptimizer(1e-1),
)

clf.train(input_fn=train_input_fn)
```

## Practical Sample Code

TODO