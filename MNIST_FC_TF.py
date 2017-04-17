import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from lib import FCLayerTF, Net

#Get MNIST data from tensorflow
dataset = tf.contrib.learn.datasets.load_dataset('mnist')
X_train = a.train.images
Y_train = a.train.labels

X_test = a.test.images
Y_test = a.test.labels
