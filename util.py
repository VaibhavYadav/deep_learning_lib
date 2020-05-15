import numpy as np

class Model():
    
    def __init__(self):
        self.computation_graph = []
        self.parameters = []
    
    def add(self, layer):
        self.computation_graph.append(layer)
        self.parameters += layer.getParams()
    
    def initializeNetwork(self):
        for f in self.computation_graph:
            if f.type == 'linear':
                weights, bias = f.getParams()
                weights.data = np.random.randn(weights.data.shape[0], weights.data.shape[1])
                bias.data = 0.
    
    def fit(self, data, target, batch_size, num_epoch, optimizer, loss_fn):
        data_gen = DataGenerator(data, target, batch_size)
        for epoch in range(num_epoch):
            loss_history = []
            for X, Y in data_gen:
                optimizer.zeroGrad()
                for f in self.computation_graph: X = f.forward(X)
                loss = loss_fn.forward(X, Y)
                grad = loss_fn.backward()
                for f in self.computation_graph[::-1]: grad = f.backward(grad)
                loss_history += [loss]
                optimizer.step()
            print("Loss at epoch = {} and loss = {}".format(epoch,np.sum(loss_history)/len(loss_history)))
                

    def predict(self, data):
        X = data 
        for f in self.computation_graph: X = f.forward(X)
        return X
# 
# 
# 
class DataGenerator():
    def __init__(self, data, target, batch_size, shuffle=True):
        self.shuffle      = shuffle
        if shuffle:
            shuffled_indices = np.random.permutation(len(data))
        else:
            shuffled_indices = range(len(data))

        self.data         = data[shuffled_indices]
        self.target       = target[shuffled_indices]
        self.batch_size   = batch_size 
        self.num_batches  = int(np.ceil(data.shape[0]/batch_size))
        self.counter      = 0


    def __iter__(self):
        return self

    def __next__(self):
        if self.counter<self.num_batches:
            batch_data = self.data[self.counter*self.batch_size:(self.counter+1)*self.batch_size]
            batch_target = self.target[self.counter*self.batch_size:(self.counter+1)*self.batch_size]
            self.counter+=1
            return batch_data,batch_target
        else:
            if self.shuffle:
                shuffled_indices = np.random.permutation(len(self.target))
            else:
                shuffled_indices = range(len(self.target))

            self.data         = self.data[shuffled_indices]
            self.target       = self.target[shuffled_indices]

            self.counter = 0
            raise StopIteration

