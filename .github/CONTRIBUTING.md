# Contributing
I encourage everybody to contribute and share their work with Keras. Have you created and trained a model with Keras? Have you ported a trained model from Caffe to Keras? This is the place to share this models to the world.

## Add a New Model to the Zoo
All the model's additions to this repository will be done by Pull Request. For each model will be required some information such as references, model definition and it's weights.

For each model proposed, all its information related must follow the following structure.
```
models
├── {models-name}
|   ├── README.md
|   ├── download_weights.sh
|   ├── {model_name}_model.json
|   ├── model.py
|   ├── examples [Optional]
|   |   └── ...
|   └── notebooks [Optional]
|       └── ...
```

### Readme
Each model should have a README.md file where a description of the model is given and also the references to the related work.

The README.md file should follow the structure available [here](README_TEMPLATE.md).


### Model as JSON
Also each model should have its definition to JSON so it would be easier for anyone to import it. To obtain this file, run the following on your python script:
```python
json_string = model.to_json()
with open('{model_name}_model.json', 'w') as f:
    f.write(json_string)
```
If the model have its own variations and want to provide all of them, please create a folder called `models` and store all the JSON definitions on this folder.

### Model weights
Another important part to define are the weights of the model. Due to the huge size of the file which stores the weights (~300MB), the weights will not be stored on the repository. On the other hand they should be self hosted by anyone and then give a script to download the weights. All the weights must be stored in `.h5` format. One recommendation is to store in Dropbox and make the file public.

The download script should be named `download_weights.sh` and should be as follows:
```bash
wget {url_to_model_weights} -O {model_name}_weights.h5
```

So for any other user who wants to work with this model and its weights, only require to run:
```bash
sh download_weights.sh
```
### Model in Python

In addition to the model's definition and its weights, it would be great to have a python script where the model has been defined. This python file can have a function defined which return the model itself. See the following example:
```python
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model
```

### Extra information
It is also recommendable to give some additional resources such as examples of model usage or python notebooks. All this resources should be stored on its corresponding directories.
