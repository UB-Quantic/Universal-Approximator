from classes.aux_functions import relu, tanh
from classes import Approximant_NN
import numpy as np
import matplotlib.pyplot as plt

layers = 3
x = np.linspace(-10, 10, 31)
f = tanh
s = 10000
np.random.seed(13) #2 buena
# Commented lines allow to represent a relu function already optimized. chi^2 for this case: 0.0004033703020408161

#params = np.array([0.5970777159072729, 1.7116080076339248, 2.3028667710500477,
#                   -1.5029590504901817, 2.0778930042466928, -1.080997255171014,
#                   1.4329789087566507, -0.7683578380074971, -1.639806210924101,
#                   0.41769829214180415, 0.3339189700737939, -12.543618534153861])

#params = params.reshape((layers, 3))

q_nn = Approximant_NN(layers, x, f)

#q_nn.update_parameters(params)

q_nn.find_optimal_parameters(batch_size=1)

q_nn.run_complete()
plt.plot(x, np.abs(q_nn.final_states[:, 1])**2)
plt.plot(x, f(x))
plt.show()


