# Keras Model Zoo
Repository to share all the models that the community has found and worked with the [Keras](https://github.com/fchollet/keras) framework. Official documentation [here](http://keras.io)

## Install

To install this package you should first download this repository and then proceed with the installation:
```bash
git clone https://github.com/albertomontesg/keras-model-zoo.git
cd keras-model-zoo
python setup.py install
```

Soon this package would be available at `pyp`.

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
```

### Models Available

At this moment the models available are:

* **VGG**: VGG16 and VGG19
* **C3D**

If you want to add any other model, check the *Contribute* section to know how to do it.

## Contribute

On `.github/CONTRIBUTION.md` there is a detailed explanation about how to contribute to this repository with new models.
Everyone is welcome and invited to participate.
