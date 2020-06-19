from classes.aux_functions import relu, tanh
from classes import Approximant_NN
import numpy as np
import matplotlib.pyplot as plt

layers = 4
x = np.linspace(-1, 1, 101)
f = relu
s = 10000
np.random.seed(4)

q_nn = Approximant_NN(layers, x, f)

# Minimization
q_nn.find_optimal_parameters(noisy=True, batch_size=.25, verbose=True)

print((q_nn._minim_function(params=q_nn.params, noisy=True)))
q_nn.run_complete(noisy=True, samples=10000)

res = np.array([q_nn.domain,
                q_nn.outcomes])
res = np.sort(res)
plt.plot(res[0], res[1])
plt.plot(res[0], f(res[0]))
plt.show()


# 0, 4