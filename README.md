# Red Dragon AI Course Advanced NLP
This repository contains the assignment as a requirement to complete the Red Dragon AI Course on Advanced NLP. There are two components to this assignment - (i) the Toxic Word Challenge, and (ii) a NLP work of our own choice. For (ii), a chatbot is trained using the [Transformer](https://arxiv.org/abs/1706.03762) network using the [movie dialogue](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) dataset. The processing of the dialogue dataset follows that of this [script](https://github.com/suriyadeepan/datasets/blob/master/seq2seq/cornell_movie_corpus/scripts/prepare_data.py) closely.

## 1. Toxic Word Challenge
The first assignement is based on the [toxic word challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). This dataset is heavily imbalanced and could contain multiple labels per comment. Since this is a binary classification problem, I applied a 1-Dimensional Convolution Layer across a window of 5 (`stride = 5`) for two times before passing the feature maps through 2 Fully-Connected layers to produce the logits. 

### 1.1 Neural Network Model
The model as returned my `toxic_model.summary()` is as follows:
```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
x_input (InputLayer)         [(None, 251)]             0
_________________________________________________________________
embedding (Embedding)        (None, 251, 32)           672448
_________________________________________________________________
conv1d (Conv1D)              (None, 124, 64)           10304
_________________________________________________________________
tf_op_layer_Relu (TensorFlow [(None, 124, 64)]         0
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 60, 128)           41088
_________________________________________________________________
tf_op_layer_Relu_1 (TensorFl [(None, 60, 128)]         0
_________________________________________________________________
flatten (Flatten)            (None, 7680)              0
_________________________________________________________________
linear1 (Dense)              (None, 128)               983168
_________________________________________________________________
linear2 (Dense)              (None, 32)                4128
_________________________________________________________________
logits (Dense)               (None, 6)                 198
_________________________________________________________________
tf_op_layer_add (TensorFlowO [(None, 6)]               0
=================================================================
Total params: 1,711,334
Trainable params: 1,711,334
Non-trainable params: 0
```
As can be observed, the model is relatively simple with about 1.7 million parameters.

### 1.2 Losses
To handle the skewed labels, we could apply either the Focal Loss, or to weigh the sigmoid loss to allow a higher loss to be assigned to positive labels. The training loss using a weight of 25.0 for positive labels yields a precision of 0.0962 and a recall of 0.8473. The tuning of the weights is provided in the table below.


<Insert training loss here.>
