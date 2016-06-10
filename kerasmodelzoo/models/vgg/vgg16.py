from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from kerasmodelzoo.utils.data import download_file, load_np_data

_VGG_16_WEIGHTS_URL = 'https://www.dropbox.com/s/u3w3ud3hlp11nwt/vgg16_weights.h5?dl=1'

def model(weights=False, summary=False):
    vgg16_model = Sequential()
    vgg16_model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    vgg16_model.add(Convolution2D(64, 3, 3, activation='relu'))
    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(64, 3, 3, activation='relu'))
    vgg16_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(128, 3, 3, activation='relu'))
    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(128, 3, 3, activation='relu'))
    vgg16_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(256, 3, 3, activation='relu'))
    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(256, 3, 3, activation='relu'))
    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(256, 3, 3, activation='relu'))
    vgg16_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg16_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg16_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg16_model.add(Flatten())
    vgg16_model.add(Dense(4096, activation='relu'))
    vgg16_model.add(Dropout(0.5))
    vgg16_model.add(Dense(4096, activation='relu'))
    vgg16_model.add(Dropout(0.5))
    vgg16_model.add(Dense(1000, activation='softmax'))

    if weights:
        filepath = download_file('vgg16_weights.h5', _VGG_16_WEIGHTS_URL)
        vgg16_model.load_weights(filepath)

    if summary:
        print(vgg16_model.summary())

    return vgg16_model

mean = load_np_data('vgg_mean.npy')
