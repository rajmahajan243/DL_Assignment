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

```
parser.add_argument('-wp' , '--wandb_project' , required = False , metavar = "" , default = 'CS6910_A1_final' , type = str , help = "Project name used to track experiments in Weights & Biases dashboard")
parser.add_argument('-we' , '--wandb_entity' , required = False , metavar = "" , default = 'rajmahajan24' , type = str , help = "Wandb Entity used to track experiments in the Weights & Biases dashboard.")
parser.add_argument('-d' , '--dataset' , required = False , metavar = "" , default = 'fashion_mnist' , type = str , choices = ["mnist","fashion_mnist"] ,help = 'choices: ["mnist", "fashion_mnist"]')
parser.add_argument('-e' , '--epochs' , required = False , metavar = "" , default = '10' , type = str , help = "Number of epochs to train neural network.")
parser.add_argument('-b' , '--batch_size' , required = False , metavar = "" , default = '64' , type = str , help = "Batch size used to train neural network.")
parser.add_argument('-l' , '--loss' , required = False , metavar = "" , default = 'cross_entropy' , type = str , choices = ["mean_squared_error", "cross_entropy"] , help = 'choices: ["mean_squared_error", "cross_entropy"]')
parser.add_argument('-o' , '--optimizer' , required = False , metavar = "" , default = 'nag' , type = str , choices = ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] , help = 'choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]')
parser.add_argument('-lr' , '--learning_rate' , required = False , metavar = "" , default = '0.1' , type = str , help = "Learning rate used to optimize model parameters")
parser.add_argument('-m' , '--momentum' , required = False , metavar = "" , default = '0.9' , type = str , help = "Momentum used by momentum and nag optimizers.")
parser.add_argument('-beta' , '--beta' , required = False , metavar = "" , default = '0.5' , type = str , help = "Beta used by rmsprop optimizer")
parser.add_argument('-beta1' , '--beta1' , required = False , metavar = "" , default = '0.5' , type = str , help = "Beta1 used by adam and nadam optimizers.")
parser.add_argument('-beta2' , '--beta2' , required = False , metavar = "" , default = '0.5' , type = str , help = "Beta2 used by adam and nadam optimizers.")
parser.add_argument('-eps' , '--epsilon' , required = False , metavar = "" , default = '0.000001' , type = str , help = "Epsilon used by optimizers.")
parser.add_argument('-w_d' , '--weight_decay' , required = False , metavar = "" , default = '0.0005' , type = str , help = "Weight decay used by optimizers.")
parser.add_argument('-w_i' , '--weight_init' , required = False , metavar = "" , default = 'Xavier' , type = str , choices = ["random", "Xavier"] , help = 'choices: ["random", "Xavier"]')
parser.add_argument('-nhl' , '--num_layers' , required = False , metavar = "" , default = '2' , type = str , help = "Number of hidden layers used in feedforward neural network.")
parser.add_argument('-sz' , '--hidden_size' , required = False , metavar = "" , default = '128' , type = str , help = "Number of hidden neurons in a feedforward layer.")
parser.add_argument('-a' , '--activation' , required = False , metavar = "" , default = 'tanh' , type = str , choices = ["identity", "sigmoid", "tanh", "ReLU"] , help = 'choices: ["identity", "sigmoid", "tanh", "ReLU"]')


```


### Model functions -

#### run_NN()
```
def run_NN():
```
This function takes the pareameters from user or sweep_config and then calls the function neural_network() with that parameters.

#### neural_network()
```
def neural_network(x_train,y_train,x_test,y_test,eta,beta_mom,beta,beta1,beta2,epsilon,activation,initializer,optimizer,batch_size,epochs,loss_fun,weight_decay,num_layers,neurons)
```
This function takes the parameters and calls the appropriate intitlizer then optimizer on which neural network is trained. Finally computers accuracy on test data. 

### Predicting accuracy
```
def image_pridiction(x_test, y_test, theta, b, num_layers,activation)
```
This function takes W and B and computes accuracy on actual classes and prints the accuracy.

### Chosing activation fucntion
```
def activation_fun(activation,a)
```

### chosing loss function
```
def loss_function(loss_fun,yhat,y)
``` 

### Forward Propogation
```
def forward_prop(activation,theta,b,num_layers,h)
```
Works imagewise ie. at time only one image will be passed 

### backpropogation
```
def backward_prop(theta,h_list,a_list,y,yhat,num_layers,batch_size,activation,loss_fun)
```
Works imagewise ie. at time only one image expected output  will be passed

### SGD
```
def SGD(theta,b,epochs,eta,beta,activation,x_train,y_train,neurons, num_layers,batch_size,loss_fun,weight_decay)
```

### momentum
```
def momentum_SGD(theta,b,epochs,eta,beta,activation,x_train,y_train,neurons, num_layers,batch_size,loss_fun,weight_decay)
```

### RMSprop
```
def RMS_SGD(theta,b,epochs,eta,beta,activation,x_train,y_train,neurons, num_layers,batch_size,loss_fun,weight_decay,epsilon)
```

### Adam
```
def adam_SGD(theta,b,epochs,eta,beta1,beta2,activation,x_train,y_train,neurons, num_layers,batch_size,loss_fun,weight_decay,epsilon)
```

### Nadam
```
def nadam(theta,b,epochs,eta,beta1,beta2,activation,x_train,y_train,neurons, num_layers,batch_size,loss_fun,weight_decay,epsilon)
```

### Nestrov
```
def NAG(theta,b,epochs,eta,beta,activation,x_train,y_train,neurons, num_layers,batch_size,loss_fun,weight_decay)
```
