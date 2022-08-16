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

class Node:
    def __init__(self, value):
	self.value = value
	self.children = []

    def backward(self):
	for child in self.children:
	    if isinstance(child, Node):
		child.backward()

    def __repr__(self):
        return "Node: {}".format(self.value)
