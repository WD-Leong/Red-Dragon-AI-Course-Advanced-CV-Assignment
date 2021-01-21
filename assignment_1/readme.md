# Red Dragon AI Advanced Computer Vision Course
## Assignment 1

This repository contains the code and write-up for the first assignment of the Advanced Computer Vision course by Red Dragon AI. The assignment requires us to classify the category of a sketch as provided by the [QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset) dataset. 

### Data
The dataset we used is the set of images provided in `.npy` format. As the official site mentioned, the images are simplified into a 28x28 grayscale bitmap which are aligned to the center of the drawing's bounding box. Some examples are shown below. The dataset in this assignment has been filtered to contain only 20 classes out of the full 345 categories, with a total of 250,000 drawings.

### Deep Learning Model
The model that was applied is a relatively simple model consisting of a 2D Convolutional Layer with 32 filters and ReLU activation, followed by a 2D Max-Pool operation, followed by another 2D Convolutional Layer with 64 filters and ReLU activation, followed by a 2D Max-Pool operation. The feature maps are then flattened and sent through a Fully-Connected Layer with 64 units and ReLU activation and finally obtaining the logits by sending it through another Fully Connected Layer with 20 units.

The `model.summary()` output is as shown below:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
img_input (InputLayer)       [(None, 28, 28, 1)]       0
_________________________________________________________________
cnn_1 (Conv2D)               (None, 26, 26, 32)        320
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0
_________________________________________________________________
cnn_2 (Conv2D)               (None, 11, 11, 64)        18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0
_________________________________________________________________
linear (Dense)               (None, 64)                102464
_________________________________________________________________
logits (Dense)               (None, 20)                1300
=================================================================
Total params: 122,580
Trainable params: 122,580
Non-trainable params: 0
_________________________________________________________________
```

