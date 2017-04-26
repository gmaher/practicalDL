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
