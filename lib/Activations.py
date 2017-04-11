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
            -x,(numpy array), size is (batch size, xdim, 1)

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
            - delta, (numpy array), size is (batch size,1,xdim)

        returns:
            gradient, (numpy array), size is (batch size,1,xdim)
        """
        return np.transpose(self.multiplier,
            axes=self.multiplier.shape[1:])*delta
