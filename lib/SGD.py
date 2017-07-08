import numpy as np

class SGD:
    def __init__(self, net, momentum=0.9):
        self.updates = [0]*len(net.layers)
        for i in range(len(self.updates)):
            self.updates[i] = []
            for w in net.layers[i].weights:
                self.updates[i].append(np.zeros_like(w))

        self.mom = momentum
        self.net = net

    def step(self, loss, x, y, learning_rate=1e-3):

        yhat = self.net.forward(x)
        l = loss.forward(y,yhat)

        delta = loss.gradient(y,yhat)
        gradients,_ = self.net.gradient(delta)

        for i in range(len(gradients)):
            g = gradients[i]
            for j in range(len(g)):
                self.updates[i][j] = learning_rate*g[j] +\
                    self.mom*self.updates[i][j]

        for i in range(len(self.net.layers)):
            for j in range(len(self.net.layers[i].weights)):
                self.net.layers[i].weights[j] -= self.updates[i][j]

        return l
