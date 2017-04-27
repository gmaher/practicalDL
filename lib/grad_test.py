from Loss import *
from Activations import *
EPS = 1e-4
mse = MSE()
x = np.random.rand(1,3)*0.5
y = np.ones_like(x)*0.75

l = mse.forward(y,x)
delta_mse = mse.gradient(y,x)
delta_num = np.zeros_like(delta_mse)
for i in range(x.shape[1]):
    x2 = x.copy()
    print x
    print x2
    x2[0,i] += EPS
    print x2
    lnew = mse.forward(y,x2)
    print lnew
    x2[0,i]-= 2*EPS
    lnew2 = mse.forward(y,x2)
    print lnew2
    d = (lnew-lnew2)/(2*EPS)
    print d
    delta_num[0,i] = d
    x2 = x.copy()

print "MSE gradient error: {}".format(np.mean(np.abs(delta_mse-delta_num)))

def activation_num_grad(act,loss,x,y):
    for i in range(x.shape[1]):
        x2 = x.copy()
        x2[0,i] += EPS
        yhat = act.forward(x2)
        lnew = mse.forward(y,yhat)

        x2[0,i]-= 2*EPS
        yhat2 = act.forward(x2)
        lnew2 = loss.forward(y,yhat2)
        d = (lnew-lnew2)/(2*EPS)
        print d
        delta_num[0,i] = d
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
