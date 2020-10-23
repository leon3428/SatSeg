import numpy as np
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import keras.backend as K
from keras.models import load_model

x_test = np.load('dataset/x_test.npy')
y_test = np.load('dataset/y_test.npy')

def pxsoftmax(z):
    z = K.exp(z)
    norm = K.sum(z, axis = 3, keepdims=True)
    norm = K.repeat_elements(norm, 3, axis=3)
    return z/norm


def dice_metric(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def weighted_crossentropy(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    dims = y_true.shape.dims
    n = dims[1]*dims[2]*dims[3]
    weights = K.constant([1.15,1.0,1.1], dtype='float32')
    weights = K.reshape(weights, (1,1,1,3))
    entropy = y_true * K.log(y_pred) * weights
    entropy = (-1.0/n) * K.sum(entropy, axis = [1,2,3])
    return entropy

def main():
    model = load_model('final_model.h5', custom_objects={'pxsoftmax': pxsoftmax, 'dice_metric': dice_metric, 'weighted_crossentropy': weighted_crossentropy})
    results = model.evaluate(x_test, y_test, batch_size = 32)
    print(model.metrics_names)
    print(results)


if __name__ == '__main__':
    main()