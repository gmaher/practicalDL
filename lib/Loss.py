import numpy as np
EPS = 1e-5
class MSE:
    def forward(self,ytrue,ypredicted):
        """
        computes mean squared error between ytrue and ypredicted
        inputs:
            - ytrue,ypredicted, (numpy array), size = nbatch x ydim
        returns:
            - loss, float, the mean squared error across all dimensions
        """
        self.loss = np.mean(np.square(ytrue-ypredicted))
        return self.loss

    def gradient(self,ytrue,ypredicted):
        """
        Computes the gradient of the mean squared error w.r.t. input
        inputs:
            - ytrue,ypredicted, (numpy array), size = nbatch x ydim
        returns:
            - grad, (numpy array), size = nbatch x ydim, delta for back prop
        """
        self.grad = -2.0/(np.prod(ytrue.shape))*(ytrue-ypredicted)
        return self.grad

class BinaryCrossEntropy:
    def forward(self, ytrue, ypredicted):
        """
        computes the binary cross entropy loss, for 1 class classification
        inputs:
            - ytrue,ypredicted, (numpy array), size = nbatch x ydim
        returns:
            - loss, float, the binary cross entropy loss across all dimensions
        """
        self.loss = -np.mean(ytrue*np.log(ypredicted+EPS)
            + (1-ytrue)*np.log(1-ypredicted+EPS))
        return self.loss

    def gradient(self, ytrue, ypredicted):
        """
        Computes the gradient of the binary cross entropy w.r.t. input
        inputs:
            - ytrue,ypredicted, (numpy array), size = nbatch x ydim
        returns:
            - grad, (numpy array), size = nbatch x ydim, delta for back prop
        """
        self.grad = -1.0/(np.prod(ytrue.shape))*((ytrue-ypredicted)
            /((ypredicted+EPS)*(1-ypredicted+EPS)))
        return self.grad

class CategoricalCrossEntropy:
    def forward(self, ytrue, ypredicted):
        """
        computes categorical cross entropy between ytrue and ypredicted
        inputs:
            - ytrue,ypredicted, (numpy array), size = nbatch x ydim
            ytrue must be one-hot vectors, ypred must be a probability distribution
        returns:
            - loss, float, the categorical cross entropy across all dimensions
        """
        Ntrain = ytrue.shape[0]
        self.inds = np.argmax(ytrue,axis=(1))
        preds = ypredicted[np.arange(0,Ntrain),self.inds]
        self.loss = -np.mean(np.log(preds+EPS))
        return self.loss
    def gradient(self, ytrue, ypredicted):
        """
        Computes the gradient of the categorical cross entropy w.r.t. input
        inputs:
            - ytrue,ypredicted, (numpy array), size = nbatch x ydim
        returns:
            - grad, (numpy array), size = nbatch x ydim, delta for back prop
        """
        Ntrain = ytrue.shape[0]
        self.grad = np.zeros_like(ytrue)
        inds = self.inds

        self.grad[np.arange(0,Ntrain),inds] =\
         -1.0/(ytrue.shape[0])*1.0/(ypredicted[np.arange(0,Ntrain),inds]+EPS)

        return self.grad

class SoftmaxCrossEntropy:
    def forward(self, ytrue, ypredicted):
        Ntrain = ytrue.shape[0]
        ymax = np.amax(ypredicted,axis=1,keepdims=True)
        e = np.exp(ypredicted-ymax)
        self.p = e/np.sum(e,axis=1,keepdims=True)
        self.inds = np.argmax(ytrue,axis=1)
        self.loss = -np.mean(np.log(self.p[np.arange(0,Ntrain),self.inds]+EPS))
        return self.loss

    def gradient(self,ytrue,ypredicted):
        Ntrain = ytrue.shape[0]
        d = self.p
        d[np.arange(0,Ntrain),self.inds] -= 1.0
        return 1.0/(ytrue.shape[0])*d.copy()
