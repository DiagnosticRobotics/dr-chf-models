# README #

This repository is the source code for the architecture of 
the machine learning models described in our paper.
Of note, given the proprietary nature of this work though, 
we were not able to provide the code for the data processing stage and feature engineering.


### Requirements
    python>=3.6
    tensorflow
    numpy
    keras
    sklearn
    xgboost
  

The creation of all used models including logistic regression, xgboost and deep learning models can be found in the "model_creation.py" file.

The sequential deep learning models including CNN and LSTM architectures with the attention mechanism can 
be found under the "deep" folder. The sequential deep learning models implement the architecture described in:

<img src="https://i.ibb.co/qN2QqRj/image-3.png" width="600">



