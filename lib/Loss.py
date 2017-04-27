import numpy as np

class MSE:
    def forward(self,ytrue,ypredicted):
        self.loss = np.mean(np.square(ytrue-ypredicted))
        return self.loss

    def gradient(self,ytrue,ypredicted):
        self.grad = -2.0/(np.prod(ytrue.shape))*(ytrue-ypredicted)
        return self.grad
#
# class BinaryCrossEntropy:
#     def forward(self, ytrue, ypredicted):
#
#     def gradient(self, ytrue, ypredicted):
#
# class CategoricalCrossEntropy:
#     def forward(self, ytrue, ypredicted):
#
#     def gradient(self, ytrue, ypredicted):
