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
Run `executer.sh`
## Verification
Results will be printed in the console, showing the predicted class along with the confidence score and confusion matrix for the input image.

---

### For More Information
For more details on the implementation and technical aspects of the project, please refer to the code comments
For any questions or issues, please feel free to open an issue in the GitHub repository.
For contributions, please submit a pull request with your proposed changes.
For general info, refer to the Introduction section above.
about.xml May be relevant for more info about the project, but it is not necessary for understanding the code or running the project.
