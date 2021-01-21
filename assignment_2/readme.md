# Red Dragon AI Advanced Computer Vision Course
## Assignment 2

This repository contains the code and write-up for the second assignment of the Advanced Computer Vision course by Red Dragon AI. For this assignment, a simplified Object Detection model is fitted on the [PASCAL Visual Object Classes (VOC)](http://host.robots.ox.ac.uk/pascal/VOC/) dataset. 

### Dataset
The VOC dataset consists of approximately 17,000 images annotated with a total of 20 object classes including Persons, Cars, Bicycles etc. Some sample images are shown in Fig. 1 below.

Sample Image 1 | Sample Image 2
:-------------:|---------------:
![voc_image_1](voc_image_1.jpg) | ![voc_image_2](voc_image_2.jpg)

Fig. 1: Sample Images from the VOC dataset.

### Object Detection Model
The Object Detection Model is a one-stage model which uses `MobileNetv2` as the backbone model, where the feature maps of the C3, C4, C5 and final layer of the backbone model is sent into through a 2D CNN layer without using advanced features like Feature Pyramid Networks (FPNs) to predict the regression offsets and the classification logits, resulting in a total of 4 anchor boxes. The weights of the output heads are not shared across the different scales. The `MobileNetv2` model is shipped as part of Tensorflow 2.0 and can be loaded directly using
```
mobilenet_v2 = tf.keras.applications.MobileNetV2(
    input_shape=(img_rows, img_cols, 3), 
    include_top=False, weights="imagenet", pooling=None)
```
and the corresponding layer of the model can be extracted via
```
x_blk5_out = \
    mobilenet_v2.get_layer("block_13_expand_relu").output
```
Finally, the model is constructed by collating the predictions of the different anchors into a list as the output.
```
x_output = [
    x_out_small, x_out_medium, 
    x_out_large, x_out_vlarge]
obj_model = tf.keras.Model(
    inputs=mobilenet_v2.input, outputs=x_output)
```
Fig. 2 shows the network architecture in greater detail.

![network_architecture](object_detection_network_architecture.jpg)

Fig. 2: Object Detection Architecture Applied.

A model summary shows that the network has approximately 2.75 million parameters. The full breakdown of the number of parameters at each layer can be found in the file `voc_model_v5_summary.txt`.

### Training the Model
The training loss uses an L1 regression loss for the bounding box regression offsets and the Focal Loss for the classification loss. To evaluate the model's performance, the images are seperated into a training (80%) and validation (20%) dataset. Fig. 3 shows model's loss as the training progresses, while Fig. 4 shows the improvement of the model's ability to detect objects of interest as training progresses, from noisy, random detections to being able to detect the first object in the image (the person) and finally being able to detect both objects in the image (the person and the horse).

Classification Loss | Regression Loss
:------------------:|:---------------:
![cls_loss](classification_loss.jpg) | ![reg_loss](regression_losses.jpg)

Fig. 3: Classification and Regression Losses as training progresses

Detection Output at Iteration 100 | Detection Output at Iteration 2500 | Detection Output at Iteration 7500
:------------------:|:---------------:|:---------------:
![output_100](https://github.com/WD-Leong/Red-Dragon-AI-Course-Advanced-CV-Assignment/blob/master/assignment_2/Results/voc_obj_detect_100.jpg) | ![output_2500](https://github.com/WD-Leong/Red-Dragon-AI-Course-Advanced-CV-Assignment/blob/master/assignment_2/Results/voc_obj_detect_2500.jpg) | ![output_7500](https://github.com/WD-Leong/Red-Dragon-AI-Course-Advanced-CV-Assignment/blob/master/assignment_2/Results/voc_obj_detect_7500.jpg)

Fig. 4: Model's ability to detect the objects of interest as training progresses

### Average Precision of the Model
Having trained the model, Table 1 shows the average precision computed for each of the class in the validation dataset. 

Class | Average Precision
:----:|:----------------:
aeroplane | 0.2381574953890692
bicycle | 0.12890989729225025
bird | 0.12893334866429035
boat | 0.033311125916055964
bottle | 0.011619462599854757
bus | 0.1656067832538421
car | 0.025314378976345406
cat | 0.3652155032026924
chair | 0.020963368696886427
cow | 0.05173020527859238
diningtable | 0.08076837499072907
dog | 0.2870961989955125
horse | 0.11204268292682927
motorbike | 0.18377871542428503
person | 0.19602099093977443
pottedplant | 0.024885251341038542
sheep | 0.03273475495697718
sofa | 0.02973437293511299
train | 0.29411764705882354
tvmonitor | 0.04444602474754659

Table 1: 
