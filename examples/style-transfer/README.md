# Neural Algorithm of Artistic Style with TensorFlow

An implementation of "[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)" with TensorFlow.

<p align = 'left'>
<img src="img/contents/tensorflow_logo.jpg" width=181>
<img src="img/styles/udnie.jpg" width=181>
<img src="img/results/tensorflow-logo_x_udnie_2000.jpg" width=181>
</p>

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

Run Python script as below:

```
python style_transfer.py --tv_weight=0.0001 \
                         --content_weight=0.07 \
                         --style_weight=0.93 \
                         --style=img/styles/udnie.jpg \
                         --summary_iterations=2 \
                         --iterations=50
```
