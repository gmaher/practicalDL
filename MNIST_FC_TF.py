import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.rcParams.update({'font.size': 24})
import matplotlib.pyplot as plt

from lib import FCLayerTF, Net, train

#Get MNIST data from tensorflow
dataset = tf.contrib.learn.datasets.load_dataset('mnist')
X_train = dataset.train.images
Y_train = dataset.train.labels
Y_train_1hot = np.eye(Y_train.max()+1)[Y_train]

X_test = dataset.test.images
Y_test = dataset.test.labels
Y_test_1hot = np.eye(Y_test.max()+1)[Y_test]

#Create neural network
layer_1 = FCLayerTF.FCLayer(shape=(X_train.shape[1],300), activation='relu')
layer_2 = FCLayerTF.FCLayer(shape=(300,100), activation='relu')
layer_3 = FCLayerTF.FCLayer(shape=(100,Y_train.max()+1), activation='linear')

net = Net.Net()
net.addLayer(layer_1)
net.addLayer(layer_2)
net.addLayer(layer_3)

x_tf = tf.placeholder(dtype=tf.float32, shape=[None, X_train.shape[1]])
y_tf = tf.placeholder(dtype=tf.float32, shape=[None, Y_train.max()+1])

yhat = net.forward(x_tf)
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=yhat,labels=y_tf))

sess = tf.Session()
output = train.train_tf(sess,loss,x_tf,y_tf,
    X_train,Y_train_1hot,X_test,Y_test_1hot, num_iter=10000, learning_rate=1e-2)

plt.figure()
plt.plot(output['train_loss'],color='r',linewidth=2,label='train loss')
plt.plot(output['test_loss'],color='g',linewidth=2,label='test loss')
plt.legend()
plt.xlabel('steps (50 iterations/step)')
plt.ylabel('loss')
lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1)
plt.grid('on')
plt.savefig('./plots/mnist_fc_tf.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')

W = sess.run(layer_1.weights[0])
pix = int(np.sqrt(X_train.shape[1]))
W = W.reshape((pix,pix,W.shape[1]))

plt.figure()
Nplots = 6
f, axarr = plt.subplots(Nplots, Nplots)
count = 0
for i in range(Nplots):
    for j in range(Nplots):
        axarr[i, j].imshow(W[:,:,count], cmap='gray')
        axarr[i,j].set_aspect('auto')
        count += 1
        # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
        plt.setp(axarr[i,j].get_xticklabels(), visible=False)
        plt.setp(axarr[i,j].get_yticklabels(), visible=False)

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('./plots/mnist_fc_tf_weights.pdf', dpi=500)
