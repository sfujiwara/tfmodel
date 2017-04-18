# Neural Algorithm of Artistic Style with TensorFlow

An implementation of "[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)" with TensorFlow.

## Requirements

* TensorFlow
* Pillow
* SciPy

## How to Run

Download the repository as below:

```
git clone https://github.com/sfujiwara/tfmodel.git
cd examples/style-transfer
```

### Run on Local

```
python style_transfer.py
```

### Run on Google Compute Engine

```
cd gce
sh create_instance.sh
```