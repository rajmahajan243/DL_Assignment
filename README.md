# CS6910 - DL_Assignment 1
***
Name -  Raj Mahajan 


Roll Number -  CS22M067
***

## Instruction to run train.py

### Objective -

The objective is to create a feedforward neural network structure that includes backpropagation and allows for the selection of available optimizers, activation functions, and loss functions. The model will be tested on the Fashion-MNIST dataset.


### Model parameters - 

optimizer (default = Nestrov)

wandbentity ( default = rajmahajan24)

wandbproject ( default = CS6910_A1_final)

dataset ( default = fashion_mnist)

epochs ( default = 10)

batch size ( default = 64 )

loss function ( default = cross entrpoy)

learning rate ( default = 0.1)

momentum ( default = 0.9)

beta ( default = 0.5)

beta1 ( default = 0.5)

beta2 ( default = 0.5)

epsilon ( default = 1e-6 )

weight decay ( default = 0.0005 )

weight intit ( default = Xavier )

num layers ( default = 2 )

hidden size ( default = 128 )

activation ( default = tanh )


### Model functions -

#### run_NN()
This function takes the pareameters from user or sweep_config and then calls the function neural_network() with that parameters.

#### neural_network()
This function takes the parameters and calls the appropriate intitlizer then optimizer on which neural network is trained. Finally computers accuracy on test data. 
