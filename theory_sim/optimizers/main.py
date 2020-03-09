from circuits import *
from optim import *
import numpy as np
import matplotlib.pyplot as plt

qubits=4
layers=1
targ_distribution = gaussian_distr(qubits)

theta = np.zeros((2 * layers + 1, qubits, 3))
theta[:, :, 0] = np.pi / 2
theta[:, :, 2] = np.pi

theta += np.random.randn(2*layers + 1, qubits, 3)
measurements = 10000
h = 0.01
a = 0.005
b1 = 0.9
b2 = 0.999

theta_new = adam_optimizer(theta, measurements, h, targ_distribution, a, b1, b2)
print(theta_new)

'''theta_new = np.array([[[1.58962466, -1.1443698, 3.67850292],
  [ 1.55706087,  0.30060152,  1.81679857],
  [ 1.55241197, -0.72778651,  3.2060929 ],
  [ 1.50435589, -0.77947604,  2.97095208]]])'''


final_prob = get_probabilities_from_measurement(theta_new, measurements)
print(cost_square(targ_distribution, final_prob))

fig, ax = plt.subplots()
ax.scatter(np.arange(0, 2**qubits), final_prob, label='Final distribution')
ax.plot(targ_distribution, label='Target distribution', color='C1')
ax.set(ylim=[0, 1])
ax.legend()

plt.show()
