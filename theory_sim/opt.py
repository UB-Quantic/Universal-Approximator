from classes.aux_functions import relu, tanh
from classes import Approximant_NN
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

layers = 4
x = np.linspace(-1, 1, 101)
f = relu
s = 10000
np.random.seed(4)

q_nn = Approximant_NN(layers, x, f)
'''
# Commented lines allow to represent a relu function already optimized. chi^2 for this case: 0.0004033703020408161
params = np.array([0.5970777159072729, 1.7116080076339248, 2.3028667710500477,
                   -1.5029590504901817, 2.0778930042466928, -1.080997255171014,
                   1.4329789087566507, -0.7683578380074971, -1.639806210924101,
                   0.41769829214180415, 0.3339189700737939, -12.543618534153861])

params = params.reshape((layers, 3))



q_nn.update_parameters(params)
print(q_nn._minim_function(params, x))
print('\n\n')
l = []
for i in range(100):
    batch = np.random.choice(x, 15, replace=False)
    l.append(q_nn._minim_function(params, batch))

l = np.array(l)
print(np.mean(l))
print(np.var(l))
'''

# Minimization
res_x = [1.4862289268771531, 0.9107912924265824, -1.8082938018165655, 2.471978757945043, -1.3455317863570297, -0.5501353837798082, -0.9274441952315735, 0.8454546419610717, 0.10354646297101461, -1.5803741376439935, 0.5673877733148988, 8.505480693741665]
q_nn.find_optimal_parameters(init_point=res_x, noisy=True, batch_size=.25, verbose=True)

print((q_nn._minim_function(params=q_nn.params, noisy=True)))
q_nn.run_complete(noisy=True, samples=10000)

res = np.array([q_nn.domain,
                q_nn.outcomes])
res = np.sort(res)
plt.plot(res[0], res[1])
plt.plot(res[0], f(res[0]))
plt.show()

opt_params = q_nn.params.copy()

# Next step: knowing more or less the landscape
# As many grid plots as layers
# We need to study these plots

def chi_one_layer(layer, opt_params):
    x = np.linspace(-np.pi, np.pi, 10)
    grid = mesh(x)
    chis = np.zeros(grid.shape[0])
    for i, p in enumerate(grid):
        chis[i] = chi_1(p, layer, opt_params)
    return chis, grid


def chi_1(new_layer_param, layer, opt_params):
    params = opt_params.copy()
    params[layer] = np.array(new_layer_param)
    chi = q_nn._minim_function(params=params)

    return chi

def mesh(x):
    mesh = []
    for x1 in x:
        for x2 in x:
            for x3 in x:
                mesh.append([x1, x2, x3])
    mesh =np.array(mesh)
    return mesh




fig = plt.figure(figsize=(30,30))
for l in range(1, 1 + layers):
    print('layer ', l)
    ax = fig.add_subplot(2, 2, l, projection='3d')
    chis, grid = chi_one_layer(l - 1, opt_params)
    h = ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2], c = chis)
    fig.colorbar(h)

plt.show()
