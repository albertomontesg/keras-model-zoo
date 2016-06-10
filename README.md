# Keras Model Zoo
Repository to share all the models that the community has found and worked with the [Keras](https://github.com/fchollet/keras) framework. Official documentation [here](http://keras.io)

## Install

To install this package you should first download this repository and then proceed with the installation:
```bash
git clone https://github.com/albertomontesg/keras-model-zoo.git
cd keras-model-zoo
python setup.py install
```
Also as a pyp package:
```bash
pip install kerasmodelzoo
```

## Usage

The usage is really easy. For each topology available you can load the model and also the mean which was trained with.

```python
from kerasmodelzoo.models.vgg import vgg16

model = vgg16.model()
mean = vgg16.mean
```

It is also possible to load the weights or print the summary of the model if you give the parameters set to True:

```python
from kerasmodelzoo.models.vgg import vgg16

model = vgg16.model(weights=True, summary=True)
mean = vgg16.mean
model.compile(loss='mse', optimizer='sgd')
X = X - mean
model.fit(X, Y)
```

## Models Available

At this moment the models available are:

### VGG


**Reference**:
```bibtex
@article{DBLP:journals/corr/SimonyanZ14a,
  author    = {Karen Simonyan and
               Andrew Zisserman},
  title     = {Very Deep Convolutional Networks for Large-Scale Image Recognition},
  journal   = {CoRR},
  volume    = {abs/1409.1556},
  year      = {2014},
  url       = {http://arxiv.org/abs/1409.1556},
  timestamp = {Wed, 01 Oct 2014 15:00:05 +0200},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/SimonyanZ14a},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```

**Framework used**: *Caffe*

**License**: *unrestricted use*

**Dataset used to train**: ILSVRC-2014


**Description**:

This is the [Keras](http://keras.io/) model of the 16-layer network used by the VGG team in the ILSVRC-2014 competition. Project [site](http://www.robots.ox.ac.uk/~vgg/research/very_deep/). Gist where the model was obtained [here](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3).

It has been obtained by directly converting the [Caffe model](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md) provived by the authors.

In the paper, the VGG-16 model is denoted as configuration `D`. It achieves 7.5% top-5 error on ILSVRC-2012-val, 7.4% top-5 error on ILSVRC-2012-test.

Please cite the paper if you use the models.


### C3D


**Reference**:

Tran, Du, et al. *"[Learning Spatiotemporal Features With 3D Convolutional Networks](http://www.cv-foundation.org/openaccess/content_iccv_2015/html/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.html)."* Proceedings of the IEEE International Conference on Computer Vision. 2015.


**Framework used**: *[C3D](https://github.com/facebook/C3D) (Caffe fork)*

**Dataset used to train**: [Sports1M](https://github.com/gtoderici/sports-1m-dataset)


**Description**:

This model was trained using a modified [version](https://github.com/facebook/C3D)
of BVLC Caffe to support 3-Dimensional Convolutional Networks.
The C3D pre-trained model provided was trained on Sports-1M dataset and can be
used to extract 3D-conv features.

Here are some results from the paper using the C3D features.

| Dataset          | UCF101 | ASLAN       | UMD-Scene | YUPENN-Scene | Object |
| ---------------- | :----: | :---------: | :-------: | :----------: | :----: |
| C3D + linear SVM | 82.3   | 78.3 (86.5) | 87.7      | 98.1         | 22.3   |

If used this model, please refer to the citations on the project [website](http://vlg.cs.dartmouth.edu/c3d/).

## Contribute

On `.github/CONTRIBUTION.md` there is a detailed explanation about how to contribute to this repository with new models.
Everyone is welcome and invited to participate.
