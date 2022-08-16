# numpy-graphical

A graph neural network implementation in numpy.

## Dependencies
* numpy

## Implementation details

### Main components
* Graph
* Node
* Tensor
* Operation
* Loss
* SGD Optimizer
* Model
* Linear Model

#### Graph class
Represents the graph that holds the data flow. On call, a new node is added to the end of the graph. The class also encapsulates the forward and backward passes.


#### Node class
Represents a node in a Graph. It defines a value, which can be set in two ways: 

    1. Using the value parameter on initialization.

    2. Using the forward pass by passing a list of arguments to it. These arguments can be of type Node or int/float. In the latter case, a new node is created with that value and added to the children of the current node. The value attribute will be equal to the output of the forward_pass method.

    3. Using the forward_pass method by passing a list of operations as arguments to it. These operations could be any custom operation, like addition or subtraction, but also any custom model, like Linear or Sequential. 

The backward pass can be called on a node. It propagates the backward pass through its children.


#### Tensor class
Represents a tensor, which is the main element that holds the data and gradients.


#### Operation class
Represents an operation in a graph. It can be any custom operation, like addition or subtraction, but also any custom model, like Linear or Sequential. The forward pass performs the operation and returns a tensor with the result. The backward pass calculates the gradient for each input of the operation using the chain rule.


#### Loss class
Represents a loss function. It can be any custom loss function, like MSE or MAE. The forward pass calculates and returns the loss value. The backward pass calculates the gradients of the loss function w.r.t. each input of the loss function using the chain rule.


#### SGD Optimizer
The optimizer class that implements stochastic gradient descent to optimize the parameters of a model.


#### Model class
The main class that encapsulates all neural network models, like Linear or Sequential. It inherits from Operation, which enables it to be a node in a Graph. Also, it inherits from Model, which enables it to be a child of any other node in a Graph.

The forward pass implements the forward pass of the model, while the backward pass implements the backward pass of the model.

Also, it implements a training loop (train_on_batch and fit), which can be used to train the model. 


#### Linear Model
A linear model that can be used in any machine learning task. It is implemented as an extension of the Model class.
