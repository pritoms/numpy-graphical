# Documentation and Logs

## Forward and Backward Pass

### Part 1. Forward Pass

To implement the forward pass we need to create a computation graph where the nodes are operations (e.g. addition, multiplication, etc.) and the edges are tensors (e.g. arrays). Each edge will have a shape and value which is computed from its connected nodes. To do this we can create a `Graph` class which will contain all of our nodes and edges. Each node will be an instance of the `Node` class. Now when we want to compute the value of a node we have to check all of its children's values and store it in the graph.

```python
class Graph:
    def __init__(self, start_node):
	self.nodes = [start_node]
	self.graph = []

    def __call__(self, *args, *kwargs):
	last_node = self.nodes[-1]
	ops = self.build_tree(*args, **kwargs)
	self.update_node(last_node, ops)
	self.update_graph()

    def forward_pass(self, *args, **kwargs):
	last_node = self.nodes[-1]
	ops = self.build_tree(*args, **kwargs)
	self.update_node(last_node, ops)
	self.update_graph()
	return self.nodes[-1].value

    def build_tree(self, *args, **kwargs):
	ops = []
	for arg in args:
	    if isinstance(arg, Node):
		ops.append(arg)
	    elif isinstance(arg, (int, float)):
		ops.append(Node(arg))
	    else:
		raise TypeError("Argument must be of type Node or int/float")

	for key, value in kwargs.items():
	    if isinstance(value, Node):
		ops.append((key, value))
	    elif isinstance(value, (int, float)):
		ops.append((key, Node(value)))
	    else:
	        raise TypeError("Argument must be of type Node or int/float")

        return ops

    def update_node(self, node, ops):
        node.children = ops

    def update_graph(self):
        self.graph = [node for node in self.nodes]

    def backward_pass(self):
        for node in reversed(self.graph):
            node.backward()

    def __repr__(self):
        return "Graph: {}".format(self.nodes)
```


### Part 2. Backward Pass

To implement the backward pass we will need to create a class for the tensors. This class will have a value and will be able to store the gradient. We will also need to create an `Operation` class which will be inherited by other operations (e.g. addition, multiplication, etc.). Each operation class will have a forward and backward method which will be called when we are doing the forward and backward pass respectively. 

```python
class Tensor:
    def __init__(self, data, requires_grad=False):
	self.data = data
	self.requires_grad = requires_grad
	self.grad = None

    def backward(self):
	if self.grad is None:
	    self.grad = np.ones_like(self.data)

	if self.requires_grad:
	    return self.grad

    def __repr__(self):
        return "Tensor: {}".format(self.data)
```


```python
class Operation:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repres__(self):
        return "Operation: {}".format(self.name)
```


We will also need to create a gradient class that will be able to take the derivative of each operation. We can do this by creating an abstract base class `Loss` that will be inherited by the different loss functions (e.g. mean squared error).

```python
class Loss(Operation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        return self.forward_pass()

    def backward(self):
        return self.backward_pass()

    def forward_pass(self):
        raise NotImplementedError

    def backward_pass(self):
        raise NotImplementedError

    def __repr__(self):
        return "Loss: {}".format(self.name)
```


## Optimizers

### Part 3. Stochastic Gradient Descent (SGD)

Now that we have our gradient class we can create an optimizer class that will be able to update the parameters based on the gradients calculated by the gradient class. To do this we will create a `SGD` class that takes in a set of parameters and a learning rate.

```python
class SGD:
    def __init__(self, parameters, lr=0.01):
	self.parameters = parameters
	self.lr = lr

    def zero_grad(self):
	for parameter in self.parameters:
	    parameter.grad = None

    def step(self):
	for parameter in self.parameters:
	    if parameter.requires_grad:
	        parameter -= self.lr * parameter.grad

    def __repr__(self):
        return "SGD: {}".format(self.lr)
```


### Part 4. Create a Model

Now that we have our gradient class, optimizer class and loss class we can create a model class. This model class will take in a set of parameters and it will need to be compiled by specifying an optimizer and a loss function. The model will also need to be able to train on a batch, fit based on an epochs, predict and evaluate.

```python
class Model:
    def __init__(self, parameters):
	self.parameters = parameters

    def zero_grad(self):
	for parameter in self.parameters:
	    parameter.grad = None

    def step(self):
	for parameter in self.parameters:
	    if parameter.requires_grad:
	        parameter -= self.lr * parameter.grad

    def __repr__(self):
        return "Model: {}".format(self.parameters)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def train_on_batch(self, x, y):
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        loss.backward()
        self.optimizer.step()
        self.zero_grad()

    def fit(self, x, y, epochs=10):
        for epoch in range(epochs):
            for x_batch, y_batch in zip(x, y):
                self.train_on_batch(x_batch, y_batch)

    def predict(self, x):
        return self.forward(x)

    def evaluate(self, x, y):
        return self.loss(self.predict(x), y).data

    def compile(self, optimizer=SGD(), loss=MSE()):
	self.optimizer = optimizer
	self.loss = loss

    def __repr__(self):
        return "Model: {}".format(self.parameters)
```


Now we can create a linear model class that will inherit from our model class. This class will take in the number of in features and out features and will create a `Linear` instance of the `Model` class.

```python
class Linear(Model):
    def __init__(self, in_features, out_features):
	super().__init__([Tensor(np.random.randn(in_features, out_features), requires_grad=True), Tensor(np.random.randn(out_features), requires_grad=True)])

    def forward(self, x):
        self.x = x

        return x @ self.parameters[0] + self.parameters[1]

    def backward(self):
        if self.x.requires_grad:
            self.x.grad += self.grad @ self.parameters[0].T

        if self.parameters[0].requires_grad:
            self.parameters[0].grad += self.x.T @ self.grad

        if self.parameters[1].requires_grad:
            self.parameters[1].grad += np.sum(self.grad, axis=0)

        return None
```


Now that we have out model class we can create a `main.py` file with some example data to test out our model.

```python
if __name__ == "__main__":
    x = Tensor(np.array([[1, 1], [0, 1], [1, 0], [0, 0]]))
    y = Tensor(np.array([[1, 0], [1, 0], [1, 0], [0, 1]]))
    model = Linear(2, 2)
    model.compile(optimizer=SGD([model.parameters[0], model.parameters[1]], lr=0.01), loss=MSE())
    model.fit(x, y)
    print(np.round(model.predict(x).data, 3))
```


## License
[MIT](https://choosealicense.com/licenses/mit/)
