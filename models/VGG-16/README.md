# VGG-16
## Information

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


## Description

This is the [Keras](http://keras.io/) model of the 16-layer network used by the VGG team in the ILSVRC-2014 competition. Project [site](http://www.robots.ox.ac.uk/~vgg/research/very_deep/). Gist where the model was obtained [here](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3).

It has been obtained by directly converting the [Caffe model](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md) provived by the authors.

In the paper, the VGG-16 model is denoted as configuration `D`. It achieves 7.5% top-5 error on ILSVRC-2012-val, 7.4% top-5 error on ILSVRC-2012-test.

Please cite the paper if you use the models.
