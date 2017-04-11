from lib import FCLayer
import numpy as np

fc = FCLayer.FCLayer((10,20),'relu')
x = np.random.randn(100,10)
delta = np.random.randn(100,20)
out = fc.forward(x)
grad = fc.gradient(delta)
