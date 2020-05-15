from base_class import Optimizer

class SGD(Optimizer):

    def __init__(self, parameters, lr=.001):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p.data = p.data - self.lr*p.grad