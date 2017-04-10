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

        self.W = np.random.randn(shape)*init
        self.b = np.random.randn(shape[1])*init

    def forward(self,x):
        """
        compute a fully-connected forward pass on x

        inputs:
            -x, (vector), 1D vector, with size self.shape[0], input to the layer

        returns:
            -out, (vector), 1D vector  with size self.shape[1], layer output
        """

        self.output = self.W.dot(x)+self.b
        return self.output

    def gradient(self,delta):
        """
        Compute the fully-connected layer gradient

        inputs:
            -delta, (vector), the upstream derivative, size is self.shape[1]

        returns:
            -grad, (tuple), tuple contain derivative wrt to W, b and x
        """
