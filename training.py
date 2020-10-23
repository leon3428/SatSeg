import numpy as np
import os
import matplotlib.pyplot as plt

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Concatenate, Input, BatchNormalization, Activation
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.optimizers import Adam
import keras.backend as K
import random
import cv2
import math
from rotate_and_crop import rotate_and_crop
from tqdm import tqdm
from keras.callbacks import LearningRateScheduler

x_train = np.load('dataset/x_train.npy')
y_train = np.load('dataset/y_train.npy')

x_val = np.load('dataset/x_val.npy')
y_val = np.load('dataset/y_val.npy')

x_test = np.load('dataset/x_test.npy')
y_test = np.load('dataset/y_test.npy')

valLoss_file = "dsboard_project/valLoss.data"
trainLoss_file = "dsboard_project/trainLoss.data"
valDice_file = "dsboard_project/valDice.data"
trainDice_file = "dsboard_project/trainDice.data"
loss_file = "dsboard_project/loss.data"
lr_file = "dsboard_project/lr.data"

INIT_LR = 1e-3
MIN_LR = 1e-5
LR_DECAY = 0.94

class LrVis(keras.callbacks.Callback):
    def __init__(self):
        self.__epoch = 0
        self.__clearFile(valLoss_file)
        self.__clearFile(trainLoss_file)
        self.__clearFile(valDice_file)
        self.__clearFile(trainDice_file)
        self.__clearFile(loss_file)
        self.__clearFile(lr_file)
        self.__batch = 0

    def __appendData(self, file, point):
        with open(file, 'a') as f:
            f.write(str(point[0]) + ',' + str(point[1]) + '/\n')

    def __clearFile(self, file):
        with open(file, 'w') as f:
            f.write('')

    def on_train_begin(self, logs={}):
        self.__epoch+=1

    def on_epoch_end(self, epoch, logs={}):
        self.__appendData(valLoss_file, (self.__epoch*2 + epoch, logs.get('val_loss')))
        self.__appendData(trainLoss_file, (self.__epoch*2 + epoch, logs.get('loss')))
        self.__appendData(valDice_file, (self.__epoch*2 + epoch, logs.get('val_dice_metric')))
        self.__appendData(trainDice_file, (self.__epoch*2 + epoch, logs.get('dice_metric')))
        self.__appendData(lr_file, (self.__epoch*2 + epoch, K.eval(self.model.optimizer.lr)))

    def on_batch_end(self, batch, logs={}):
        self.__appendData(loss_file, (self.__batch, logs.get('loss')))
        self.__batch += 1
    

def schedule(epoch, lr):
    lr *= LR_DECAY
    return max(lr, MIN_LR)

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

def conv_block(inp, filters):
    conv1 = Conv2D(filters, (3,3), activation = 'relu', padding = 'same')(inp)
    conv2 = Conv2D(filters, (3,3), activation = 'relu', padding = 'same')(conv1)

    return conv2

def build_model():
    inputs = Input(shape=(128,128,4))
    en_block1 = conv_block(inputs, 16)
    en_pool1 = MaxPool2D((2,2))(en_block1)
    en_block2 = conv_block(en_pool1, 32)
    en_pool2 = MaxPool2D((2,2))(en_block2)
    en_block3 = conv_block(en_pool2, 64)
    en_pool3 = MaxPool2D((2,2))(en_block3)
    en_block4 = conv_block(en_pool3, 128)
    en_pool4 = MaxPool2D((2,2))(en_block4)

    en_block5 = conv_block(en_pool4, 256)

    
    de_upconv1 = Conv2DTranspose(128, (3,3), strides = (2,2), activation = 'relu', padding = 'same')(en_block5)
    de_concat1 = Concatenate()([de_upconv1, en_block4])
    de_block1 = conv_block(de_concat1, 128)
    
    de_upconv2 = Conv2DTranspose(64, (3,3), strides = (2,2), activation = 'relu', padding = 'same')(de_block1)
    de_concat2 = Concatenate()([de_upconv2, en_block3])
    de_block2 = conv_block(de_concat2, 64)

    de_upconv3 = Conv2DTranspose(32, (3,3), strides = (2,2), activation = 'relu', padding = 'same')(de_block2)
    de_concat3 = Concatenate()([de_upconv3, en_block2])
    de_block3 = conv_block(de_concat3, 32)

    de_upconv4 = Conv2DTranspose(16, (3,3), strides = (2,2), activation = 'relu', padding = 'same')(de_block3)
    de_concat4 = Concatenate()([de_upconv4, en_block1])
    de_block4 = conv_block(de_concat4, 16)

    output = Conv2D(3, (1,1), activation = pxsoftmax, padding = 'same')(de_block4)
    
    ret = Model(inputs = inputs, outputs = output, name = 'unet')
    ret.compile(
        optimizer = Adam(lr = INIT_LR),
        loss = weighted_crossentropy,
        metrics=[dice_metric]
        )

    return ret


def data_augmentation():
    x_aug = []
    y_aug = []
    rotations = [cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180]

    for i in tqdm(range(len(x_train))):
        x = x_train[i]
        y = y_train[i]

        rows,cols, ch = x.shape
        
        flip = random.randrange(-1, 1)
        x = cv2.flip(x, flip)
        y = cv2.flip(y, flip)

        angle = random.randrange(0,3)
        if angle != 3:
            x = cv2.rotate(x, rotations[angle])
            y = cv2.rotate(y, rotations[angle])
        
        
        brightness = random.randrange(-10,10)/100
        x += brightness

        r_shift = random.randrange(95,105)/100
        g_shift = random.randrange(95,105)/100
        b_shift = random.randrange(95,105)/100
        i_shift = random.randrange(95,105)/100
        x[:,:,0]*=b_shift
        x[:,:,1]*=g_shift
        x[:,:,2]*=r_shift
        x[:,:,3]*=i_shift

        if (128,128) != x.shape[:2]:
            x = cv2.resize(x, (128, 128), interpolation = cv2.INTER_AREA)
        if (128,128) != y.shape[:2]:
            y = cv2.resize(y, (128, 128), interpolation = cv2.INTER_AREA)

        x_aug.append(x)
        y_aug.append(y)

    return (np.array(x_aug), np.array(y_aug))



def main():
    print(x_train.shape, x_val.shape, x_test.shape)
    print(y_train.shape, y_val.shape, y_test.shape)

    model = build_model()
    #model = load_model('model-02-0.89.h5', custom_objects={'pxsoftmax': pxsoftmax, 'dice_metric': dice_metric, 'weighted_crossentropy': weighted_crossentropy})
    plot_model(model, "dsboard_project/model.png", show_shapes=True)

    
    filepath="batchnorm/model-{epoch:02d}-{val_dice_metric:.3f}.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor='val_dice_metric',
        mode='max')

    learning_vis =  LrVis()
    lr_sch = LearningRateScheduler(schedule, verbose=0)

    for i in range(45):
        print(f'Epoch: {i} and {i+1}')
        print('Augmenting data')
        x,y = data_augmentation()
        model.fit(x, y, batch_size=32, epochs=2, callbacks=[learning_vis, checkpoint_callback, lr_sch], validation_data=(x_val, y_val))
    

if __name__ == '__main__':
    main()