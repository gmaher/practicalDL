class ReLU:
    def forward(self,x):
        self.multiplier = np.zeros(x.shape)

        self.multiplier[x>0] = 1

        return self.multiplier*x

    def gradient(self,delta):

        return np.transpose(self.multiplier,
            axes=self.multiplier.shape[1:])*delta
