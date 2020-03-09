from classes import *
import numpy as np
import matplotlib.pyplot as plt

layers = 3
x = np.linspace(-1, 1, 20)
f = relu
s = 10000

q_nn = Approximant_NN(layers, x, f)
q_nn.find_optimal_parameters(batch=1)



#plt.ion()
plt.show()
#plt.draw()
#plt.pause(5)

