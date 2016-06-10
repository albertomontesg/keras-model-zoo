from keras.layers.convolutional import (Convolution3D, MaxPooling3D,
                                        ZeroPadding3D)
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from kerasmodelzoo.utils.data import download_file, load_np_data

_C3D_WEIGHTS_URL = 'https://www.dropbox.com/s/ypiwalgtlrtnw8b/c3d-sports1M_weights.h5?dl=1'

def model(weights=False, summary=True):
    c3d_model = Sequential()
    # 1st layer group
    c3d_model.add(Convolution3D(64, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv1',
                            subsample=(1, 1, 1),
                            input_shape=(3, 16, 112, 112)))
    c3d_model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='valid', name='pool1'))

    # 2nd layer group
    c3d_model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv2',
                            subsample=(1, 1, 1)))
    c3d_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool2'))

    # 3rd layer group
    c3d_model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3a',
                            subsample=(1, 1, 1)))
    c3d_model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3b',
                            subsample=(1, 1, 1)))
    c3d_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool3'))

    # 4th layer group
    c3d_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4a',
                            subsample=(1, 1, 1)))
    c3d_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4b',
                            subsample=(1, 1, 1)))
    c3d_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool4'))

    # 5th layer group
    c3d_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5a',
                            subsample=(1, 1, 1)))
    c3d_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5b',
                            subsample=(1, 1, 1)))
    c3d_model.add(ZeroPadding3D(padding=(0, 1, 1)))
    c3d_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool5'))
    c3d_model.add(Flatten())

    # FC layers group
    c3d_model.add(Dense(4096, activation='relu', name='fc6'))
    c3d_model.add(Dropout(.5))
    c3d_model.add(Dense(4096, activation='relu', name='fc7'))
    c3d_model.add(Dropout(.5))
    c3d_model.add(Dense(487, activation='softmax', name='fc8'))

    if weights:
        filepath = download_file('c3d_weights.h5',
            _C3D_WEIGHTS_URL)
        c3d_model.load_weights(filepath)

    if summary:
        print(c3d_model.summary())

    return c3d_model

mean = load_np_data('c3d_mean.npy')
