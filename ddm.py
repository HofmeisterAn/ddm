#!/usr/bin/env pytho

'''This module create a keras machine learning model.'''

import argparse
import json
import numpy

# Third party frameworks and classes
from keras import backend as K
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD

# Custom classes
from plot import Plot
from image import ImageData

PLOT = Plot()

TRAIN_IMAGES = []

TEST_IMAGES = []

MODEL_WEIGHT_PATH = 'model.h5'

IMAGE_WIDTH, IMAGE_HEIGHT, BATCH_SIZE = 255, 255, 32

def load_data():
    data = json.load(open('data/metadata.json'))

    for image in data['images']:
        file_path = image['filePath']
        x = image['x']
        y = image['y']

        image = ImageData(file_path, IMAGE_WIDTH, IMAGE_HEIGHT, x, y)

        if 'train' in file_path:
            TRAIN_IMAGES.append(image)
        else:
            TEST_IMAGES.append(image)

def train_data_count():
    return len(TRAIN_IMAGES)

def test_data_count():
    return len(TEST_IMAGES)

def get_train_data():
    return TRAIN_IMAGES

def get_test_data():
    return TEST_IMAGES

def reshape_data(data):
    data_x = []
    data_y = []

    for image in data:
        if not numpy.any(data_x) and not numpy.any(data_y):
            data_x = image.get_data_x()
            data_y = image.get_data_y()
        else:
            data_x = numpy.concatenate((data_x, image.get_data_x()), axis=0)
            data_y = numpy.concatenate((data_y, image.get_data_y()), axis=0)

    return data_x, data_y

def get_image_shape():
    if K.image_data_format() == 'channels_first':
        return (1, IMAGE_WIDTH, IMAGE_HEIGHT)
    else:
        return (IMAGE_WIDTH, IMAGE_HEIGHT, 1)

def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(get_image_shape())))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dense(3))

    sgd = SGD(lr=0.03, momentum=0.9, nesterov=True)

    model.compile(loss='mse', optimizer=sgd)

    return model

def generate_model(save_history, save_model):
    if save_history is None:
        save_history = True

    if save_model is None:
        save_model = False

    # Loads the data from data/metadata.json
    load_data()

    train_x, train_y = reshape_data(get_train_data())
    test_x, test_y = reshape_data(get_test_data())

    model = cnn_model()

    hist = model.fit(train_x, train_y, batch_size=1, epochs=25, validation_data=(test_x, test_y))

    if save_history:
        PLOT.save_loss(hist.history['loss'])
        PLOT.save_loss_validation(hist.history['val_loss'])

    if save_model:
        model.save_weights(MODEL_WEIGHT_PATH, overwrite=True)

def validate_model():
    # Loads the data from data/metadata.json
    load_data()

    model = cnn_model()
    model.load_weights(MODEL_WEIGHT_PATH)

    for data in get_test_data():
        test_x, test_y = reshape_data([data])
        score = model.predict(test_x)
        print(data.get_file_path())
        print(data.get_points())
        print(score)

def plot():
    PLOT.plot_loss()

def main():
    parser = argparse.ArgumentParser()

    sub_parser = parser.add_subparsers(dest='subparser')
    sub_parser.required = True

    # Enable generate model sub task
    parser_a = sub_parser.add_parser('generate_model')
    parser_a.add_argument('--saveHistory', dest='save_history', help='True saves the history values in cvs files.')
    parser_a.add_argument('--saveModel', dest='save_model', help='True saves the model.')

    # Enable validate model sub task
    sub_parser.add_parser('validate_model')

    # Enable plot sub task
    sub_parser.add_parser('plot')

    kwargs = vars(parser.parse_args())
    globals()[kwargs.pop('subparser')](**kwargs)

if __name__ == '__main__':
    main()
