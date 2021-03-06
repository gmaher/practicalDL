import numpy as np

class Net:
    def __init__(self):
        self.layers = []

    def addLayer(self,layer):
        self.layers.append(layer)

    def forward(self,x, N=None):
        o = self.layers[0].forward(x)

        if N == None:
            N = len(self.layers)

        for i in range(1,N):
            o = self.layers[i].forward(o)

        return o

    def gradient(self, delta):
        gradients = [0]*len(self.layers)

        for i in range(len(self.layers)-1,-1,-1):
            layer_gradients,delta = self.layers[i].gradient(delta)
            gradients[i] = layer_gradients

        return gradients,delta
