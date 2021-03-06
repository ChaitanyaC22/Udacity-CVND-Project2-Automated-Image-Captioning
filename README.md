# Udacity-CVND-Project2-Automated-Image-Captioning

## Objective
This project aims at training a CNN-RNN model to predict captions for a given image. The main task is to implement an effective RNN decoder for a CNN encoder.

[![image](https://github.com/ChaitanyaC22/Udacity-CVND-Project2-Automated-Image-Captioning/blob/chai_main/images/image_captioning_model.png)](https://github.com/ChaitanyaC22/Udacity-CVND-Project2-Automated-Image-Captioning/blob/chai_main/images/image_captioning_model.png)<br>

## Project Overview<br>
The goal of this project is to create a neural network architecture to automatically generate captions from images. Please checkout `requirements.txt` for the necessary packages required. <br><br>**Important:** `Pytorch version 0.4.0` required.

The Microsoft Common Objects in COntext [(MS COCO) dataset](http://cocodataset.org/#home) is used to train the neural network. The final model is then tested on novel images!

## Project Instructions

The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order:

-   0\_Dataset.ipynb
-   1\_Preliminaries.ipynb
-   2\_Training.ipynb
-   3\_Inference.ipynb and <br>
`model.py`: Network Architecture. <br>

## Network Architecture

The network architecture consists of:

1.  The CNN encoder converts images into embedded feature vectors: [![image](https://github.com/ChaitanyaC22/Udacity-CVND-Project2-Automated-Image-Captioning/blob/chai_main/images/encoder.png)](https://github.com/ChaitanyaC22/Udacity-CVND-Project2-Automated-Image-Captioning/blob/chai_main/images/encoder.png)
2.  The feature vector is translated into a sequence of tokens by an RNN Decoder, which is a sequential neural network made up of LSTM units: [![image](https://github.com/ChaitanyaC22/Udacity-CVND-Project2-Automated-Image-Captioning/blob/chai_main/images/decoder.png)](https://github.com/ChaitanyaC22/Udacity-CVND-Project2-Automated-Image-Captioning/blob/chai_main/images/decoder.png)

## Results

These are some of the outputs/captions generated by the neural network on a couple of test images from test data of [COCO dataset](http://cocodataset.org/): <br><br>
[![output1](https://github.com/ChaitanyaC22/Udacity-CVND-Project2-Automated-Image-Captioning/blob/chai_main/images/output1.png)](https://github.com/ChaitanyaC22/Udacity-CVND-Project2-Automated-Image-Captioning/blob/chai_main/images/output1.png) <br>
<br>
[![output2](https://github.com/ChaitanyaC22/Udacity-CVND-Project2-Automated-Image-Captioning/blob/chai_main/images/output2.png)](https://github.com/ChaitanyaC22/Udacity-CVND-Project2-Automated-Image-Captioning/blob/chai_main/images/output2.png)
