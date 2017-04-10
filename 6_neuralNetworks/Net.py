import numpy as np

class Net:
    def __init__(self):
        self.layers = []

    def addLayer(self,layer):
        self.layers.append(layer)

    def forward(self,x):
        o = self.layers[0].forward(x)

        for i in range(1,len(self.layers)):
            o = self.layers[i].forward(o)

        return o
