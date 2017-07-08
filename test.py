from lib import FCLayer,Net,Loss
import numpy as np

fc = FCLayer.FCLayer((10,20),'relu')
fc2 = FCLayer.FCLayer((20,2),'sigmoid')
fc3 = FCLayer.FCLayer((2,4),'softmax')
x = np.random.randn(100,10)
y = np.ones((100,4))
delta = np.random.randn(100,20)
out = fc.forward(x)
grad = fc.gradient(delta)

net = Net.Net()
net.addLayer(fc)
net.addLayer(fc2)
net.addLayer(fc3)
yhat = net.forward(x)

loss = Loss.MSE()
mse = loss.forward(y,yhat)
delta = loss.gradient(y,yhat)

gradients,delta = net.gradient(delta)

def numerical_gradient(layer,loss,x,y):
    W = layer.weights[0]
    b = layer.weights[1]
    W_new = W.copy()

    yhat = layer.forward(x)
    l = loss.forward(y,yhat)

    delta = loss.gradient(y,yhat)
    grad,d2 = layer.gradient(delta)
    num_grad = np.zeros_like(W)

    I,J = W.shape
    for i in range(I):
        for j in range(J):
            W_new[i,j] += 1e-4
            layer.weights[0] = W_new
            ynew = layer.forward(x)
            loss_new = loss.forward(y,ynew)
            d = (loss_new-l)/1e-4
            num_grad[i,j] = d
            W_new = W.copy()

    return num_grad.copy(), np.mean(np.abs(num_grad-grad[0]))

x1 = np.random.rand(20,10)
x2 = np.random.rand(20,20)
x3 = np.random.rand(20,2)

y1 = np.random.rand(20,20)
y2 = np.random.rand(20,2)
y3 = np.random.rand(20,4)
a1= numerical_gradient(fc, loss, x1,y1)
print a1[1]
a2= numerical_gradient(fc2, loss, x2,y2)
print a2[1]
a3= numerical_gradient(fc3, loss, x3,y3)
print a3[1]
