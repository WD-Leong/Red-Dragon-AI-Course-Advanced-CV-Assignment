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
The Object Detection Model is a lightweight model which uses `MobileNetv2` as the backbone model, with the feature maps corresponding to their respective blocks being sent into a 2D CNN without using advanced features like Feature Pyramid Networks (FPNs). A total of 4 anchor boxes per scale was used.
