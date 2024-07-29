# Object Detection with MobileNet SSD V1

## Table of Contents
- [Introduction](#introduction)
- [Object Detection vs Image Classification](#object-detection-vs-image-classification)
- [Computer Vision and TensorFlow](#computer-vision-and-tensorflow)
- [Object Detection Models](#object-detection-models)
- [Labeling Images for Training](#labeling-images-for-training)
- [Fine-Tuning the Model](#fine-tuning-the-model)
- [Project Directory Structure](#project-directory-structure)
- [Web Application Development with Flask](#web-application-development-with-flask)
- [Acknowledgements](#acknowledgements)
- [References](#references)

## Introduction

Object detection is a crucial task in computer vision that involves identifying and locating objects within an image or video frame. This project focuses on using MobileNet SSD V1, a lightweight and efficient deep learning model designed for object detection tasks on resource-constrained devices such as mobile phones and embedded systems.

## Object Detection vs Image Classification

### Object Detection
Object detection identifies and locates objects within an image. It provides both the class of the object and the bounding box coordinates around the object.

### Image Classification
Image classification assigns a label to an entire image, identifying what object or scene is present in the image without providing location information.

## Computer Vision and TensorFlow

### Computer Vision
Computer vision is a field of artificial intelligence that enables computers to interpret and make decisions based on visual data from the world.

### TensorFlow
TensorFlow is an open-source machine learning framework developed by Google. We use TensorFlow v1.x for this project due to its compatibility with the MobileNet SSD V1 model.

## Object Detection Models

### YOLO (You Only Look Once)
YOLO is a real-time object detection system known for its speed and accuracy.

### EfficientDet
EfficientDet is an object detection model that balances efficiency and accuracy using compound scaling.

### MobileNet SSD V1
MobileNet SSD V1 is a lightweight model that provides a good balance between speed and accuracy, making it suitable for mobile and embedded devices.

## Labeling Images for Training

Labeling images involves annotating images with the correct bounding boxes and labels for the objects of interest. We use the `labelImg` tool for this purpose.

## Fine-Tuning the Model

### Fine-Tuning and Transfer Learning
Fine-tuning involves taking a pre-trained model and retraining it on a new dataset to adapt it to a specific task. Transfer learning leverages pre-trained models to reduce the training time and improve performance.

## Important Requirements

Below are the main libraries and their versions required for this project:

- **TensorFlow**: 1.15.0
- **Flask**: 2.2.5
- **LabelImg**: Latest
- **OpenCV**: 4.6.0.66
- **h5py**: 3.8.0
- **JupyterLab**: 3.5.3

You can install these requirements using the following command:

pip install tensorflow==1.15.0 flask==2.2.5 opencv-python==4.6.0.66 h5py==3.8.0 jupyterlab==3.5.3

## Project Directory Structure

Here is the directory structure for the project:

### Steps and File Names
1. **Data Collection**: Collect images and store them in the `uploads` directory.
2. **Data Labeling**: Use `labelImg` to annotate images.
3. **Model Training**: Train the model using TensorFlow.
4. **Model Fine-Tuning**: Fine-tune the pre-trained MobileNet SSD V1.
5. **Model Conversion**: Convert the trained model to TensorFlow Lite format.
6. **Web Application**: Create a web application using Flask.

## Web Application Development with Flask

Flask is a lightweight WSGI web application framework in Python. We use Flask to develop a web application that utilizes the trained object detection model.

### Web Application Overview
The web application allows users to upload images, and it performs object detection on the uploaded images, displaying the results with bounding boxes and labels.

## Acknowledgements

I would like to express my special thanks to my SABUDH mentors Mrs. Alka Yadav and Mr. Ravi Mittal for their supervision and guidance. Their valuable input has greatly contributed to the success of this project. I would also like to thank my college professors and my parents and friends for their support and encouragement.

## References

- TensorFlow: https://www.tensorflow.org/
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- LabelImg: https://github.com/tzutalin/labelImg
- Labelme: https://github.com/wkentaro/labelme
