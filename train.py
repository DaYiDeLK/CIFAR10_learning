# -*- coding: utf-8 -*-
# @Time    : 2017/12/15 13:52
# @File    : train.py
# @Author  : Rock

import time
import models
import util
import os
from keras.datasets import cifar10
from keras.utils import to_categorical

# normalization
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.
x_test = x_test / 255.

# convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

input_shape = x_train.shape[1:]

# print data set information
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

batch_size = 32
num_classes = 10
nb_epoch = 100

model = models.cnn01(input_shape)
start = time.time()
h = util.adam_op(model, x_train, y_train, x_test, y_test, nb_epoch, batch_size)

# 将模型保存到以下目录
model_dir = 'S:/CIFAR10_MODELS/model01'
if os.path.isdir(model_dir):
    pass
else:
    os.makedirs(model_dir)

model.save(model_dir + '/CIFAR10_MODEL.h5')

# 打印训练完花的时间
print('@ Total Time Spent: %.2f seconds' % (time.time() - start))

# 调用一开始定义的函数
util.plot_acc_loss(h, nb_epoch)
util.visualize_cifar10(model, x_test, y_test)

# 打印模型训练完成后在总的训练集和测试集上的loss和accuracy
loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy = %.2f %%     loss = %f" % (accuracy * 100, loss))
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy = %.2f %%    loss = %f" % (accuracy * 100, loss))
