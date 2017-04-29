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
        print self.multiplier[0]
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

class Softmax:
    def forward(self,x):
        """
        Computes forward pass of Softmax activation

        inputs:
            -x,(numpy array), size is (batch size, xdim)

        returns:
            -output, (numpy array),1/(1+e^{-x}) applied to x, same dimensions
        """

        xmax = np.amax(x, axis=1, keepdims=True)
        e = np.exp(x-xmax)
        self.output = e/(np.sum(e, axis=1, keepdims=True))
        return self.output

    def gradient(self,delta):
        """
        calculate softmax gradient

        inputs:
            - delta, (numpy array), size is (batch size,xdim)

        returns:
            gradient, (numpy array), size is (batch size,xdim)
        """
        out = self.output
        batch_size,out_dim = out.shape
        dy = np.zeros((batch_size,out_dim,out_dim))
        diags = range(out_dim)
        for i in range(batch_size):
            dy[i] = (-out[i,:, np.newaxis]).dot(out[i,np.newaxis,:])

        dy[:,diags,diags] = out*(1-out)
        d = np.zeros((batch_size,out_dim))
        for i in range(batch_size):
            d[i] = delta[i].dot(dy[i])
        return d
