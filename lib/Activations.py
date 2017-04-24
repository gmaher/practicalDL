import numpy as np

class Linear:
    def forward(self,x):
        return x
    def gradient(self, delta):
        return delta

class ReLU:
    def forward(self,x):
        """
        Computes forward pass of ReLU activation

        inputs:
            -x,(numpy array), size is (batch size, xdim)

        returns:
            -output, (numpy array), max(0,x) applied to x, same dimensions
        """
        self.multiplier = np.zeros(x.shape)

        self.multiplier[x>0] = 1

        return self.multiplier*x

    def gradient(self,delta):
        """
        calculate ReLU gradient

        inputs:
            - delta, (numpy array), size is (batch_size,xdim)

        returns:
            gradient, (numpy array), size is (batch size,xdim)
        """
        out = self.multiplier*delta
        return self.multiplier*delta

class Sigmoid:
    def forward(self,x):
        """
        Computes forward pass of Sigmoid activation

        inputs:
            -x,(numpy array), size is (batch size, xdim)

        returns:
            -output, (numpy array),1/(1+e^{-x}) applied to x, same dimensions
        """

        self.output = 1.0/(1+np.exp(-x))
        return self.output

    def gradient(self,delta):
        """
        calculate ReLU gradient

        inputs:
            - delta, (numpy array), size is (batch size,xdim)

        returns:
            gradient, (numpy array), size is (batch size,xdim)
        """
        out = delta*self.output*(1.0-self.output)
        return delta*self.output*(1.0-self.output)
