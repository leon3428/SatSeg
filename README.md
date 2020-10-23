# SatSeg

The goal of the project was to develop a semantic segmentation algorithm that could classify satellite imagery into nature, water, and urban classes. I originally collected the dataset for the Copernicus Hackathon Zagreb 2020, but this is a more advanced version that I did not have time to implement before. 

## Dataset
The dataset contains 5483 labeled satellite pictures of size 128x128 pixels. Each image consists of four spectral bands (red, green, blue, and near-infrared). Labels are bgr images where blue stands for water, green for nature, and red for urban. The dataset is split into train(5483 images), test(550 images), and validation(550 images) sets. The quality of the dataset is pretty poor and that provided some challenges as you will see below.

## Solution
A neural network is used to achieve this task, specifically a U-net.
![model](https://raw.githubusercontent.com/leon3428/SatSeg/master/model.png)

The last layer activation is a pixel-wise softmax. Weighted categorical cross-entropy is used as the loss function. For evaluating the model, the soft dice(f1) metric is used. Data augmentation was used to combat overfitting. 

## Results
I was able to achieve 0.883 soft dice(f1) score on the test data. Considering the size of the dataset and quality of it, I am pretty happy with the results.

For larger areas like the ones below, the kernel was applied to each 128x128 section. Each kernel was then cropped by 14 pixels from each side to get rid of the lower quality predictions on the edges. On the resulting image, minimum postprocessing was applied to illustrate raw results from the network. In the images below you are seeing an input image, the segmentation output, and the output overlayed on top of the input.

![example1](https://github.com/leon3428/SatSeg/blob/master/lowResExamples/1-min.png)
![example1](https://github.com/leon3428/SatSeg/blob/master/lowResExamples/2-min.png)
![example1](https://github.com/leon3428/SatSeg/blob/master/lowResExamples/3-min.png)
![example1](https://github.com/leon3428/SatSeg/blob/master/lowResExamples/4-min.png)
![example1](https://github.com/leon3428/SatSeg/blob/master/lowResExamples/5-min.png)

As you can see the network struggles with dirt paths probably because of the datasets quality, narrow water, and rocky areas that were not present in the dataset.

## Installation
1) Clone and unzip the repo
2) Create a virtualenv inside of the project folder
```
python -m venv venv
```
3) Install keras with plaidml backend and set it up
```
pip install plaidml keras
plaidml-setup
```
4) Install other supporting libraries
```
pip install numpy, opencv-python, tqdm, matplotlib
```
