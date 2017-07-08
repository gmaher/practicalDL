import numpy as np

def circle_data(N=1000, num_classes=5, noise=0.03):
    X = np.zeros((N*num_classes,2))
    Y = np.zeros((N*num_classes,num_classes))

    for i in range(num_classes):
        r = np.random.rand(N)*2*np.pi
        eps = np.random.randn(N,2)*noise
        X[i*N:(i+1)*N,0] = np.cos(r)*(i+1.0)/num_classes+eps[:,0]
        X[i*N:(i+1)*N,1] = np.sin(r)*(i+1.0)/num_classes+eps[:,1]
        Y[i*N:(i+1)*N,i] = 1

    return (X,Y)
