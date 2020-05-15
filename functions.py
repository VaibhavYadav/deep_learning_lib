from base_class import Function, Tensor
import numpy as np

class Linear(Function):

    def __init__(self, in_nodes, out_nodes):
        self.weights = Tensor((in_nodes, out_nodes))
        self.bias = Tensor((1, out_nodes))
        self.type = 'linear'

    def forward(self, x):
        # [[1, 2]](1 X 2) * [[w11, w12, w13], [w21, w22, w23]](2 X 3) + [[b1, b2, b3]](1 X 3)
        output = np.dot(x, self.weights.data) + self.bias.data
        self.input = x
        return output

    def backward(self, d_y):
        # [[1, 2]](1 X 2) -> [[1], [2]](2 X 1) * [[1, 2, 3]](1 X 3)
        self.weights.grad += np.dot(self.input.T, d_y)
        # [[1, 2, 3], [1, 2, 3]](2 X 3) -> [[1, 2, 3]](1 X 3)
        self.bias.grad += np.sum(d_y, axis=0, keepdims=True)
        # [[1, 2, 3]](1 X 3) * [[w11, w21], [w12, w22], [w13, w23]](3 X 2)
        grad_input = np.dot(d_y, self.weights.data.T)
        return grad_input

    def getParams(self):
        return [self.weights, self.bias]


class Sigmoid(Function):

    def __init__(self):
        self.type = 'activation'

    def sigmoid_function(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_prime_function(self, x):
        return self.sigmoid_function(x)*(1 - self.sigmoid_function(x))

    def forward(self, x):
        self.input = x
        return self.sigmoid_function(x)

    def backward(self, d_y):
        return d_y*self.sigmoid_prime_function(self.input)


class SquareLoss(Function):
    def __init__(self):
        self.type = 'loss'

    def forward(self, x, y):
        self.x = x
        self.y = y 
        return np.sum(x - y)**2

    def backward(self):
        grad = (self.x - self.y)
        return grad
