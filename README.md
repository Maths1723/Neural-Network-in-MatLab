# Neural Network Implementation in MATLAB

This repository contains a basic implementation of a neural network using MATLAB for binary classification tasks. It was made on Dec-2024 and Jan-2025 for a university project. The assignement forced to implement the Neural Network from Scratch and to implement a simple Stochastic Gradient Descent (SGD) with Mean Squared Error (MSE) as the loss function. Here's an overview of what you'll find in this project:

## Overview

- **Network Architecture**: A fully connected (feedforward) neural network with customizable layers.
- **Learning Algorithm**: Stochastic Gradient Descent (SGD) with Mean Squared Error (MSE) as the loss function.
- **Activation Function**: Sigmoid for each neuron.

## File Structure

- `main.m`: Contains the main script that sets up the data, constructs the network, and trains it.
- `train.m`: Function for training the neural network using SGD.
- `sigmoid.m`, `sigmoidGradient.m`, `randInitializeWeights.m`, `predict.m`: Helper functions for network operations.

## Setup

### Input Data

X, where each data is a column vector. Ex: X=[x1',x2',...,xn'].  
Y, labels, again as where each label is a column vector.  


### Training parameters setup
alfa = 5e-2;    % Learning rate  
maxIter = 1000000;  % Maximum number of iterations  
batch_size = 1;  % Batch size for SGD (set to 1 for stochastic training)


## Visualization

The visualization component of this MATLAB script provides real-time feedback on the training process of the neural network, however this might slow down the SGD computing, consider changing the "25" in the following line accordingly.

```matlab
    % Visualization update every 25 iterations
    if mod(i,25) == 0
