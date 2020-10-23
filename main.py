import numpy as np
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import keras.backend as K
from keras.models import load_model
import cv2
from tqdm import tqdm

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

def post_processing(img):
    binary = np.zeros_like(img, dtype='float32')
    idx = np.argmax(img, axis=2)
    idx = np.expand_dims(idx, 2)
    np.put_along_axis(binary, idx, 1, axis=2)

    return binary

def load_test_image(ind):
    ret = np.zeros((5000,5000,4), dtype='float32')
    lines = ['b2', 'b3', 'b4', 'b8']
    for i in range(4):
        im_name = 'test_images/im' + str(ind) + '_' + lines[i] + '.bmp'
        ch = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)
        ret[:,:,i] = ch

    return ret

def slice_img(src, overlap):
    ret = []
    src /= 255
    h,w,ch = src.shape
    for i in range(0,h,128-2*overlap):
        for j in range(0,w,128-2*overlap):
            if i+128 >= h or j+128 >= w:
                continue

            img = src[i:i+128, j:j+128, :]
            ret.append(img)

    return np.array(ret)

def tile(h, w, tiles, overlap):
    ret = np.ones((h,w,3))
    br = 0
    tSize = 128-2*overlap
    for i in range(0,h,128-2*overlap):
        for j in range(0,w,128-2*overlap):
            if i+128 >= h or j+128 >= w:
                continue

            processed = post_processing(tiles[br])

            istart = i+overlap-1
            jstart = j+overlap-1
            ret[istart:istart+tSize, jstart:jstart+tSize, : ] = processed[overlap:128-overlap, overlap:128-overlap, :]
            br+=1

    return ret

def main():
   
    model = load_model('final_model.h5', custom_objects={'pxsoftmax': pxsoftmax, 'dice_metric': dice_metric, 'weighted_crossentropy': weighted_crossentropy})


    for k in tqdm(range(1,7)):
        test_img = load_test_image(k)

        nn_input = slice_img(test_img, 14)
        nn_output = model.predict(nn_input)

        h,w,ch = test_img.shape
        sol = tile(h,w,nn_output, 14)
        
        
        test_img = test_img[: , :, :3]
        test_img*=255
        test_img = np.uint8(test_img)
        sol*=255
        sol = np.uint8(sol)
        test_img = test_img[: , :, :3]

        alpha = 0.7
        overlay = cv2.addWeighted(test_img, alpha, sol, 1-alpha, 0.0)
        
        cv2.imwrite(f'output_images/{k}in.png', test_img)
        cv2.imwrite(f'output_images/{k}out.png', sol)
        cv2.imwrite(f'output_images/{k}over.png', overlay)


if __name__ == '__main__':
    main()