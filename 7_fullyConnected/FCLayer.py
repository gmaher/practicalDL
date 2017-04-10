import numpy as np

class FCLayer:
    def __init__(self,shape,activation,init=1e-3):
        """
        initializer for a fully-connected layer

        inputs:
            -shape, (tuple), input,output size of layer
            -activation, (string), activation function to use
            -init, (float), multiplier for random weight initialization
        """
        self.shape = shape
        self.activation = activation

        self.weights = []
        self.weights.append(np.random.randn(shape)*init)
        self.weights.append(np.random.randn(shape[1])*init)

        if activation == 'relu':
            self.activation = ReLU()

    def forward(self,x):
        """
        compute a fully-connected forward pass on x

        inputs:
            -x, (numpy array), with size (batch size, self.shape[0])
                , input to the layer

        returns:
            -out, (numpy array),  with size (batch size, self.shape[1]),
            layer output
        """

        self.h = self.weights[0].dot(x)+self.weights[1]
        self.a = self.activation.forward(self.h)
        return self.a

    def gradient(self,delta):
        """
        Compute the fully-connected layer gradient

        inputs:
            -delta, (vector), the upstream derivative, size is self.shape[1]

        returns:
            -grad, (tuple), tuple contain derivative wrt to W, b and x
        """
