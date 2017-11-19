# Handwriting-Number-Recognition
Takes 20x20 handwritten images as input and uses one-vs-all regularized logistic regression in order to train a neural network to discern each number (0-9)

The Recognition.m file is the main file that performs the image recognition when run.
The ThetaWeights.mat file is a text file containing all the weights used in the neural network.
The TrainingData.mat file contains all the observation images used in training the neural network.
The predict.m function 

The sigmoid.m file contains the sigmoid function to calculate the costs (training SSE) of each observation when inputted into the model. It also contains a gradient-based algorithm used to minimize the costs and optimize the neural network weights.
