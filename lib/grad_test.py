from Loss import *
from Activations import *
import FCLayer
import SGD
import Net

EPS = 1e-4
mse = MSE()
x = np.random.rand(2,3)*0.5
y = np.zeros_like(x)
y[:,1] = 1.0

def loss_num_grad(loss,x,y):
    delta_num = np.zeros_like(y)
    l = loss.forward(y,x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x2 = x.copy()
            x2[i,j] += EPS
            lnew = loss.forward(y,x2)
            x2[i,j]-= 2*EPS
            lnew2 = loss.forward(y,x2)
            d = (lnew-lnew2)/(2*EPS)
            delta_num[i,j] = d
            x2 = x.copy()
    return delta_num.copy()

delta_mse = mse.gradient(y,x)
delta_num = loss_num_grad(mse,x,y)
print "MSE gradient error: {}".format(np.mean(np.abs(delta_mse-delta_num)))

bce = BinaryCrossEntropy()
delta_bce = bce.gradient(y,x)
bce_num = loss_num_grad(bce,x,y)
print "binary cross entropy gradient error: {}".format(np.mean(np.abs(delta_bce-bce_num)))

cce = CategoricalCrossEntropy()
l = cce.forward(y,x)
delta_cce = cce.gradient(y,x)
cce_num = loss_num_grad(cce,x,y)
print "CategoricalCrossEntropy gradient error: {}".format(np.mean(np.abs(delta_cce-cce_num)))

def activation_num_grad(act,loss,x,y):
    delta_num = np.zeros_like(y)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x2 = x.copy()
            x2[i,j] += EPS
            yhat = act.forward(x2)
            lnew = mse.forward(y,yhat)

            x2[i,j]-= 2*EPS
            yhat2 = act.forward(x2)
            lnew2 = loss.forward(y,yhat2)
            d = (lnew-lnew2)/(2*EPS)
            delta_num[i,j] = d
            x2 = x.copy()

    return delta_num.copy()

linear = Linear()
yhat = linear.forward(x)
delta = mse.gradient(y,yhat)
grad = linear.gradient(delta)
grad_num = activation_num_grad(linear,mse,x,y)
print "linear activation gradient error: {}".format(np.mean(np.abs(grad-grad_num)))

linear = Sigmoid()
yhat = linear.forward(x)
delta = mse.gradient(y,yhat)
grad = linear.gradient(delta)
grad_num = activation_num_grad(linear,mse,x,y)
print "Sigmoid activation gradient error: {}".format(np.mean(np.abs(grad-grad_num)))

linear = Softmax()
yhat = linear.forward(x)
delta = mse.gradient(y,yhat)
grad = linear.gradient(delta)
grad_num = activation_num_grad(linear,mse,x,y)
print "Softmax activation gradient error: {}".format(np.mean(np.abs(grad-grad_num)))

#Test fully-connected layer gradient
fc = FCLayer.FCLayer((x.shape[1],y.shape[1]),'softmax')

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
            W_new[i,j] += EPS
            layer.weights[0] = W_new
            ynew = layer.forward(x)
            loss_new = loss.forward(y,ynew)
            d = (loss_new-l)/EPS
            num_grad[i,j] = d
            W_new = W.copy()

    b_grad = np.zeros_like(layer.weights[1])
    b_new = b.copy()
    for i in range(b_grad.shape[1]):
        b_new[0,i] += EPS
        layer.weights[1] = b_new
        ynew = layer.forward(x)
        loss_new = loss.forward(y,ynew)
        b_new[0,i] -= 2*EPS
        layer.weights[1] = b_new
        ynew = layer.forward(x)
        loss_new2 = loss.forward(y,ynew)
        d = (loss_new-loss_new2)/(2*EPS)
        b_grad[0,i] = d
        b_new = b.copy()

    delta_num = np.zeros_like(y)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x2 = x.copy()
            x2[i,j] += EPS
            yhat = layer.forward(x2)
            lnew = loss.forward(y,yhat)

            x2[i,j]-= 2*EPS
            yhat2 = layer.forward(x2)
            lnew2 = loss.forward(y,yhat2)
            d = (lnew-lnew2)/(2*EPS)
            delta_num[i,j] = d
            x2 = x.copy()

    return num_grad.copy(), b_grad.copy(), delta_num.copy()

num_grad, num_b, num_x = numerical_gradient(fc, mse, x, y)
yhat = fc.forward(x)
delta = mse.gradient(y,yhat)
grad, dx = fc.gradient(delta)
dw = grad[0]
db = grad[1]
print "Fully-connect layer gradient error = {}".format(np.mean(np.abs(num_grad-dw)))
print "Fully-connect layer gradient error b = {}".format(np.mean(np.abs(num_b-db)))
print "Fully-connect layer gradient error x = {}".format(np.mean(np.abs(num_x-dx)))

#Test optimizers
fc = FCLayer.FCLayer((10,20),'relu')
fc2 = FCLayer.FCLayer((20,2),'sigmoid')
fc3 = FCLayer.FCLayer((2,4),'softmax')
x = np.random.randn(100,10)
y = np.zeros((100,4))
inds = np.random.randint(4,size=100)
y[:,inds] = 1.0

net = Net.Net()
net.addLayer(fc)
net.addLayer(fc2)
net.addLayer(fc3)

sgd = SGD.SGD(net,momentum=0.9)
for i in range(10000):
    l = sgd.step(net,cce,x,y, learning_rate=1e-1)
    if i%1000==0:
        print 'SGD iteration {}: loss={}'.format(i,l)
