# -*- coding: utf-8 -*-
# @Time    : 2017/12/16 10:37
# @File    : util.py
# @Author  : Rock

import keras
import numpy as np
import matplotlib.pyplot as plt


# visualize cifar10
def visualize_cifar10(model, x_test, y_test):
    class_name = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck',
    }

    rand_id = np.random.choice(range(10000), size=10)
    y_true = [y_test[i] for i in rand_id]
    y_true = np.argmax(y_true, axis=1)
    y_true = [class_name[name] for name in y_true]

    x_pred = np.array([x_test[i] for i in rand_id])
    y_pred = model.predict(x_pred)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = [class_name[name] for name in y_pred]
    plt.figure(figsize=(15, 7))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_pred[i].reshape(32, 32, 3), cmap='gray')
        plt.title('True: %s \n Pred: %s' % (y_true[i], y_pred[i]), size=15)
    plt.show()


def plot_acc_loss(h, nb_epoch):
    acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(range(nb_epoch), acc, label='Train')
    plt.plot(range(nb_epoch), val_acc, label='Test')
    plt.title('Accuracy over ' + str(nb_epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(nb_epoch), loss, label='Train')
    plt.plot(range(nb_epoch), val_loss, label='Test')
    plt.title('Loss over ' + str(nb_epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.show()


def adam_op(model, x_train, y_train, x_test, y_test, nb_epoch, batch_size):
    adam = keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    h = model.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size, validation_data=(x_test, y_test),
                  shuffle=True, verbose=2)
    return h
