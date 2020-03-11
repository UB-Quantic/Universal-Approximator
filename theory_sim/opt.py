from classes.aux_functions import relu
from classes import Approximant_NN
import numpy as np
import matplotlib.pyplot as plt

layers = 3
x = np.linspace(-1, 1, 101)
f = relu
s = 10000
params = np.array([-0.9239235180266095, -0.8103263939122924, 0.00034424080300026697, 1.7694393927681662, 1.743440044322635, 0.05274748286820163, 0.7562270690836563, -0.14785030405003585, -1.2641893649988525])
params = params.reshape((layers, 3))
q_nn = Approximant_NN(layers, x, f)
#q_nn.update_parameters(params)

'''chi = q_nn._minim_function(params, batch=x)
a=[]
print(chi)
for i in range(1000):
    batch = np.random.choice(x, 20, replace=False)
    chi = q_nn._minim_function(params, batch=batch)
    a.append(chi)
a = np.array(a)
print(np.mean(a))
print(np.var(a))'''
q_nn.find_optimal_parameters(batch_size=.1)
# plt.ion()
# plt.show()
# plt.draw()
# plt.pause(5)

