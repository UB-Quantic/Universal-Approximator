from classes import *
import numpy as np
import matplotlib.pyplot as plt

layers = 3
x = np.linspace(-1, 1, 50)
f = relu
s = 10000
#np.random.seed()
q_nn = Approximant_NN(layers, x, f)
q_nn.find_optimal_parameters(noisy=False, batch=.5)



# plt.ion()
# plt.show()
# plt.draw()
# plt.pause(5)

