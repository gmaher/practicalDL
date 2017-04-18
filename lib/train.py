import numpy as np
import tensorflow as tf


def get_batch(xdata, ydata, nbatch):
    N = len(ydata)
    inds = np.random.choice(N, size=nbatch, replace=True)
    xret = xdata[inds,:]
    if len(ydata.shape) == 1:
        yret = ydata[inds]
    else:
        yret = ydata[inds,:]
    return (xret,yret)

def train_tf(sess,loss,x_tensor,y_tensor,X,Y,X_test,Y_test,
    opt='sgd', learning_rate=1e-3,num_iter=1000, batch_size=64):
    """
    wrapper for training with tensorflow

    -inputs:
        sess, (tensorflow session)
        loss, (tensorflow tensor), the loss tensor we want to minimize
        x_tensor, (tensorflow tensor), placeholder tensor for x
        y_tensor, (tensorflow tensor), placeholder tensor for y
        X, (numpy array), training input data
        Y, (numpy array), training labels
        X_test, (numpy array), validation input data
        Y_test, (numpy array), validation labels
        opt, (string), optimizer to use (sgd or adam),
        num_iter, (int), number of training batches to go through
        batch_size, (int), number of examples per batch

    -returns:
        output, (dictionary), dictionary containing training history
    """

    if opt == 'sgd':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
            momentum=0.9)
    if opt == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
            momentum=0.9)

    train = optimizer.minimize(loss)
    sess.run(tf.initialize_all_variables())

    output = {}
    output['train_loss'] = []
    output['test_loss'] = []
    for i in range(num_iter):
        x,y = get_batch(X,Y,batch_size)
        sess.run(train,{x_tensor:x,y_tensor:y})

        if i%50 == 0:
            l_train = sess.run(loss,{x_tensor:x,y_tensor:y})
            l_test = sess.run(loss,{x_tensor:X_test,y_tensor:Y_test})
            output['train_loss'].append(l_train)
            output['test_loss'].append(l_test)
            print "iteration {}: train loss={}, test loss={}".format(i,l_train,l_test)
    return output
