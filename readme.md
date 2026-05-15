# AI Image Classifier
This is a CPU optimized AI object detector and image classifer. for more info, read the Introduction section below, For technical details, please refer to the code comments in `optimizedcputrainer.py` and `classifyandpredict.py`.
## Introduction
This project is an AI image classifier that uses a convolutional neural network (CNN) to classify images into different categories.
The project is implemented in Python using PyTorch.
It is CPU optimized, making it suitable for training and inference on machines without a CUDA GPU.

---

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Platform](https://img.shields.io/badge/Platform-CPU--Only-lightgrey)
![Model](https://img.shields.io/badge/Model-MobileNetV2-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/License-UNLICENSED-red)

---

## Installation
Follow these steps to install the project:
1. To install the required dependencies, run the following command:
`pip install -r requirements.txt`
2. Make sure you have Python 3.8 or higher installed on your machine.
3. Ensure that you have the necessary permissions to read and write files in the project directory.
4. Clone the repository to your local machine using the following command:
`git clone https://github.com/jrf-g/PythonApplicationArtificialIntelligencelmageClassifier`
5. Navigate to the project directory:
`cd PythonApplicationArtificialIntelligencelmageClassifier`
## Setup
1. write class names into `samples/classnames.json`
2. put training images into `samples`
3. put image to classify into `classify.png`
## Execution
Follow these steps to execute the project:
1. run `optimizedcputrainer.py` to train the model and save the trained model as `image_classifier_mobilenetv2_cpu.pth`
2. run `classifyandpredict.py`
Or just run `executer.sh` to execute both steps in one go.
## Verification
Results will be printed in the console, showing the predicted class along with the confidence score and confusion matrix for the input image.