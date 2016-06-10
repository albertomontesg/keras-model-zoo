# Contributing
I encourage everybody to contribute and share their work with Keras. Have you created and trained a
model with Keras? Have you ported a trained model from Caffe to Keras? This is the place to share
this models to the world.

## Add a New Model to the Zoo
All the model's additions to this repository will be done by Pull Request. For each model will be
required some information such as references, model definition and it's weights and means.

As this repository is a python package, new models should be added as new modules on
`kerasmodelzoo/models`.

If the new model has multiple version create a folder with the generic name and subfiles for each
version of the model. If, on the other hand, it has only one version, create a new file with its
name.
```
kerasmodelzoo
├── models
|   ├── __init__.py
|   ├── modelA
|   |   ├── __init__.py
|   |   ├── modelA_v1.py
|   |   └── modelA_v2.py
|   ├── modelB.py
```

### Model File

For each model file this variables and functions should be given.
```python

from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from kerasmodelzoo.utils.data import download_file, load_np_data

# URL to download the weights
_MODEL_WEIGHTS_URL = 'https://www.url.edu/where/model/is/stored.hdf5

def model(weights=False, summary=False):
    vgg16_model = Sequential()
    '''
    Here comes the definition of the model's architecture such as:

    vgg16_model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    vgg16_model.add(Convolution2D(64, 3, 3, activation='relu'))
    '''

    # This lines are required to load the weights if they are asked
    if weights:
        filepath = download_file('vgg16_weights.h5', _VGG_16_WEIGHTS_URL)
        vgg16_model.load_weights(filepath)

    if summary:
        print(vgg16_model.summary())

# A variable with the dataset mean the model was trained with.
mean = load_np_data('{mean_file}.npy')
```

The mean file should be placed at the `kerasmodelzoo/data/{mean_file}.py`.

The weights should be stored in a hdf5 file without being compressed. It must be stored on Keras.

### Extra information

It is also recommendable to give some additional resources such as examples of model usage or python
notebooks. All this resources should be stored on its corresponding directories.
