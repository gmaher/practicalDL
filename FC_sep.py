import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.rcParams.update({'font.size': 24})
import matplotlib.pyplot as plt

from lib import FCLayer, Net, train, Loss
from lib import data

num_classes = 3
N = 1000
colors = ['r','b','g','k','y']
symbols = ['o','x','D','*','v']
labels = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5']
X_train, Y_train_1hot = data.circle_data(N=N,num_classes=num_classes)

#Create neural network
layer_1 = FCLayer.FCLayer(shape=(X_train.shape[1],5), activation='sigmoid',init=1.0)
layer_2 = FCLayer.FCLayer(shape=(5,2), activation='sigmoid',init=1.0)
layer_3 = FCLayer.FCLayer(shape=(2,2), activation='sigmoid',init=1.0)
layer_4 = FCLayer.FCLayer(shape=(2,Y_train_1hot.shape[1]), activation='softmax',init=1.0)

net = Net.Net()
net.addLayer(layer_1)
net.addLayer(layer_2)
net.addLayer(layer_3)
net.addLayer(layer_4)

#Set up loss function and train
loss = Loss.CategoricalCrossEntropy()
output = train.train(net,loss,
    X_train,Y_train_1hot,X_train,Y_train_1hot, num_iter=6000, learning_rate=1e-1)

g1 = net.forward(X_train,N=2)
g2 = net.forward(X_train,N=3)
plt.figure()
for i in range(num_classes):
    plt.scatter(X_train[i*N:(i+1)*N,0],X_train[i*N:(i+1)*N,1],color=colors[i],marker=symbols[i],
    label=labels[i])
plt.grid('on')
plt.xlabel('x1')
plt.ylabel('x2')
lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1)
plt.grid('on')
plt.savefig('./plots/circle_fc_1.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')


plt.figure()
for i in range(num_classes):
    plt.scatter(g1[i*N:(i+1)*N,0],g1[i*N:(i+1)*N,1],color=colors[i],marker=symbols[i],
    label=labels[i])
plt.grid('on')
plt.xlabel('x1')
plt.ylabel('x2')
lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1)
plt.grid('on')
plt.savefig('./plots/circle_fc_2.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.figure()
for i in range(num_classes):
    plt.scatter(g2[i*N:(i+1)*N,0],g2[i*N:(i+1)*N,1],color=colors[i],marker=symbols[i],
    label=labels[i])
plt.grid('on')
plt.xlabel('x1')
plt.ylabel('x2')
lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1)
plt.grid('on')
plt.savefig('./plots/circle_fc_3.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')
