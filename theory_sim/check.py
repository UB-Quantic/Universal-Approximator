from classes import *
import numpy as np
import matplotlib.pyplot as plt

layers = 3
x = np.linspace(-5, 5, 20)
f = tanh
s = 10000


q_nn = Approximant_NN(layers, x, f)

params = [0.36792659006488304, 1.586745058085428, -1.361994401766359, -0.734699127833574, 0.006460943806065676, 0.8919101771072313, 0.7284647110096059, -0.010319639066890587, 22.693416102123226]
params = np.asarray(params)
params = params.reshape((3, layers))

q_nn.update_parameters(params)
q_nn.run(noisy=True, samples=s)

plt.plot(x, f(x))
plt.plot(x, q_nn.outcomes)

params = [-4.452983727600961e-05, 1.1896292822844698, -1.0639031435264825, 0.9418060342369164, -2.451508924822802, -1.8916155114145803, 0.47079849770069565, -3.141551740101474, -1.057952143235473]
params = np.asarray(params)
params = params.reshape((3, layers))

q_nn.update_parameters(params)
q_nn.run()
plt.plot(x, np.abs(q_nn.final_states[:, 1])**2)
plt.show()
