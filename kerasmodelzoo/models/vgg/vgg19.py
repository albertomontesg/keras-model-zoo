from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from kerasmodelzoo.utils.data import download_file, load_np_data

_VGG_19_WEIGHTS_URL = 'http://files.heuritech.com/weights/vgg19_weights.h5'

def model(weights=False, summary=False):
    vgg19_model = Sequential()

    vgg19_model.add(ZeroPadding2D((1, 1),input_shape=(3, 224, 224)))
    vgg19_model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    vgg19_model.add(ZeroPadding2D((1, 1)))
    vgg19_model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    vgg19_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg19_model.add(ZeroPadding2D((1, 1)))
    vgg19_model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    vgg19_model.add(ZeroPadding2D((1, 1)))
    vgg19_model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    vgg19_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg19_model.add(ZeroPadding2D((1, 1)))
    vgg19_model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    vgg19_model.add(ZeroPadding2D((1, 1)))
    vgg19_model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    vgg19_model.add(ZeroPadding2D((1, 1)))
    vgg19_model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    vgg19_model.add(ZeroPadding2D((1, 1)))
    vgg19_model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_4'))
    vgg19_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg19_model.add(ZeroPadding2D((1, 1)))
    vgg19_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    vgg19_model.add(ZeroPadding2D((1, 1)))
    vgg19_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    vgg19_model.add(ZeroPadding2D((1, 1)))
    vgg19_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    vgg19_model.add(ZeroPadding2D((1, 1)))
    vgg19_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_4'))
    vgg19_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg19_model.add(ZeroPadding2D((1, 1)))
    vgg19_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    vgg19_model.add(ZeroPadding2D((1, 1)))
    vgg19_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    vgg19_model.add(ZeroPadding2D((1, 1)))
    vgg19_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    vgg19_model.add(ZeroPadding2D((1, 1)))
    vgg19_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_4'))
    vgg19_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg19_model.add(Flatten())
    vgg19_model.add(Dense(4096, activation='relu', name='dense_1'))
    vgg19_model.add(Dropout(0.5))
    vgg19_model.add(Dense(4096, activation='relu', name='dense_2'))
    vgg19_model.add(Dropout(0.5))
    vgg19_model.add(Dense(1000, name='dense_3'))
    vgg19_model.add(Activation("softmax"))

    if weights:
        filepath = download_file('vgg19_weights.h5',
            _VGG_19_WEIGHTS_URL)
        vgg19_model.load_weights(filepath)

    if summary:
        print(vgg19_model.summary())

    return vgg19_model

mean = load_np_data('vgg_mean.npy')
