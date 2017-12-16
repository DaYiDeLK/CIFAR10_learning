# -*- coding: utf-8 -*-
# @Time    : 2017/12/16 10:37
# @File    : models.py
# @Author  : Rock
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D


# model01
def cnn01(input_shape):
    model = Sequential()
    # conv1
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding="same", input_shape=input_shape))
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Activation('relu'))

    # con2
    model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same'))
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Activation('relu'))

    # conv3
    model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same'))
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Activation('relu'))

    # hidden
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))

    # output
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model
