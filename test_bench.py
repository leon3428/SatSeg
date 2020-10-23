import numpy as np
import os
import matplotlib.pyplot as plt

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
from keras.models import load_model
import cv2
import keras.backend as K

def pxsoftmax(z):
    z = K.exp(z)
    norm = K.sum(z, axis = 3, keepdims=True)
    norm = K.repeat_elements(norm, 3, axis=3)
    return z/norm

def dice_metric(y_true, y_pred, smooth=0.5):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def weighted_crossentropy(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    dims = y_true.shape.dims
    n = dims[1]*dims[2]*dims[3]
    weights = K.constant([1.1,1.0,1.1], dtype='float32')
    weights = K.reshape(weights, (1,1,1,3))
    entropy = y_true * K.log(y_pred) * weights
    entropy = (-1.0/n) * K.sum(entropy, axis = [1,2,3])
    return entropy

x_val = np.load('dataset/x_val.npy')
y_val = np.load('dataset/y_val.npy')

cv2.namedWindow('x', cv2.WINDOW_NORMAL)
cv2.namedWindow('y', cv2.WINDOW_NORMAL)

model = load_model('batchnorm/model-02-0.887.h5', custom_objects={'pxsoftmax': pxsoftmax, 'dice_metric': dice_metric, 'weighted_crossentropy': weighted_crossentropy})
#model = load_model('ch30.h5')

for i in range(30, 50):
    x = x_val[i]
    nn_in = np.expand_dims(x, axis=0)
    
    y = model.predict(nn_in)

    cv2.imshow('x', x)
    cv2.imshow('y', y[0])
    cv2.waitKey(0)