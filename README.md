# Handwritten-Digit-Recognition
Takes 20x20 handwritten images as input and uses one-vs-all regularized logistic regression in order to train a neural network to discern each number (0-9). 
10-fold Cross validation is used on various models to determine the best decay parameters and number of hidden layers. 

The Recognition.m file is the main file that performs the image recognition when run.
The ThetaWeights.mat file is a text file containing all the weights used in the neural network.
The TrainingData.mat file contains all the observation images used in training the neural network.
The predict.m function takes in an observation and returns the predicted value based on the neural network.

The sigmoid.m file contains the sigmoid function to calculate the costs (training SSE) of each observation when placed as input into the model.

The backpropagation folder contains all the files used to perform backpropagation that optimized the weights of the model.
