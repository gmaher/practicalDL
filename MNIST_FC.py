import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.rcParams.update({'font.size': 24})
import matplotlib.pyplot as plt

from lib import FCLayer, Net, train, Loss

#Get MNIST data from tensorflow
dataset = tf.contrib.learn.datasets.load_dataset('mnist')
X_train = dataset.train.images
Y_train = dataset.train.labels
Y_train_1hot = np.eye(Y_train.max()+1)[Y_train]

X_test = dataset.test.images
Y_test = dataset.test.labels
Y_test_1hot = np.eye(Y_test.max()+1)[Y_test]

#Create neural network
layer_1 = FCLayer.FCLayer(shape=(X_train.shape[1],300), activation='relu')
layer_2 = FCLayer.FCLayer(shape=(300,100), activation='relu')
layer_3 = FCLayer.FCLayer(shape=(100,Y_train.max()+1), activation='softmax')

net = Net.Net()
net.addLayer(layer_1)
net.addLayer(layer_2)
net.addLayer(layer_3)

#Set up loss function and train
loss = Loss.CategoricalCrossEntropy()
output = train.train(net,loss,
    X_train,Y_train_1hot,X_test,Y_test_1hot, num_iter=10000, learning_rate=1e-1)

plt.figure()
plt.plot(output['train_loss'],color='r',linewidth=2,label='train loss')
plt.plot(output['test_loss'],color='g',linewidth=2,label='test loss')
plt.legend()
plt.xlabel('steps (50 iterations/step)')
plt.ylabel('loss')
lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1)
plt.grid('on')
plt.savefig('./plots/mnist_fc.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')

W = layer_1.weights[0]
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
plt.savefig('./plots/mnist_fc_weights.pdf', dpi=500)
