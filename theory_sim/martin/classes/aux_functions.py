import numpy as np
import os
import json

# Collections of functions to approximate

def step(x):
    step.name = 'step'
    return 0.5 * (np.sign(x) + 1)

def cosine(x):
    cosine.name = 'cosine'
    return 0.5 * (np.cos(2*np.pi*x) + 1)

def sigmoid(x):
    sigmoid.name = 'sigmoid'

    return 1 / (1 + np.exp(-x))

def tanh(x):
    tanh.name = 'tanh'

    return 0.5 * (np.tanh(x) + 1)

def relu(x):
    relu.name = 'relu'

    return np.clip(x, 0, np.max(x))

def poly(x):
    poly.name= 'poly'

    return np.abs(3*x**3 * (1 - x**4))


def M3(x, params_3):
    params_3 = 0.5 * params_3
    m = np.array([[np.exp(1j * params_3[2]) * np.cos(params_3[0] * x + params_3[1]), -np.exp(1j * params_3[2]) * np.sin(params_3[0] * x + params_3[1])],
                  [np.exp(-1j * params_3[2]) * np.sin(params_3[0] * x + params_3[1]), np.exp(-1j * params_3[2]) * np.cos(params_3[0] * x + params_3[1])]],
                 dtype='complex')
    return m

def M4(x, params_4):
    params_4 = 0.5 * params_4
    m = np.array([[np.exp(1j * (params_4[2] * x + params_4[3])) * np.cos(params_4[0] * x + params_4[1]),
                   -np.exp(1j * (params_4[2] * x + params_4[3])) * np.sin(params_4[0] * x + params_4[1])],
                  [np.exp(-1j * (params_4[2] * x + params_4[3])) * np.sin(params_4[0] * x + params_4[1]),
                   np.exp(-1j * (params_4[2] * x + params_4[3])) * np.cos(params_4[0] * x + params_4[1])]],
                 dtype='complex')
    return m

def M5(x, params_5):
    params_5 = 0.5 * params_5
    m = np.array([[np.cos(params_5[4]) * np.cos(params_5[3]) * np.exp(1j * (params_5[1] + params_5[0] * x))
                   - np.sin(params_5[4]) * np.sin(params_5[3]) * np.exp(1j * (params_5[2] - params_5[0] * x)),
                   -np.cos(params_5[4]) * np.sin(params_5[3]) * np.exp(1j * (params_5[1] + params_5[0] * x))
                   - np.sin(params_5[4]) * np.cos(params_5[3]) * np.exp(1j * (params_5[2] - params_5[0] * x))],
                  [np.sin(params_5[4]) * np.cos(params_5[3]) * np.exp(1j * (-params_5[2] + params_5[0] * x))
                   + np.cos(params_5[4]) * np.sin(params_5[3]) * np.exp(1j * (-params_5[1] - params_5[0] * x)),
                   - np.sin(params_5[4]) * np.sin(params_5[3]) * np.exp(1j * (-params_5[2] + params_5[0] * x))
                   + np.cos(params_5[4]) * np.cos(params_5[3]) * np.exp(1j * (-params_5[1] - params_5[0] * x))]],
                 dtype='complex')

    return m

def state(x, theta):
    """
    Core function of the algorithm. This function computes the quantum circuits
    :param x: x values, independent variables
    :param theta: parameters defining the quantum circuit
    :return: output wavefunction
    """
    params=theta.shape[-1]
    if params == 3:
        M = M3(x, theta[0, :])
        for i in range(1, theta.shape[0]):
            M = M3(x, theta[i, :]) @ M

    if params == 4:
        M = M4(x, theta[0, :])
        for i in range(1, theta.shape[0]):
            M = M4(x, theta[i, :]) @ M

    if params == 5:
        M = M5(x, theta[0, :])
        for i in range(1, theta.shape[0]):
            M = M3(x, theta[i, :]) @ M
    s = np.array([[1],[0]])
    s = M @ s

    return s

# Previous trials
'''def minimize_with_previous_result(algorithm, layers, domain, f, phase=None, gens=100, tol=1e-8):
    try:
        alg = algorithm(layers, domain, f)
    except:
        alg = algorithm(layers, domain, f, phase)
    filename = 'results/' + alg.name + '/' + alg.f.name + '/%s_exact.txt' % (alg.layers - 1)
    with open(filename, 'r') as outfile:
        prev_result = json.load(outfile)

    if not prev_result['success']:
        raise ImportError('Previous result did not converge')

    shape = alg.params.shape

    init_point = np.asfarray(prev_result['x']).reshape((shape[0] - 1, shape[1]))
    init_point = np.concatenate((np.zeros((1, 3)), init_point), axis=0)
    alg.update_parameters(init_point)
    res_q = alg.find_optimal_parameters(gens=gens, tol=tol)
    print(res_q)'''

def fold_name(appr_name, f):
    folder = 'results/' + appr_name + '/' + f.name
    return folder

def save_dict(result, folder, filename):
    try:
        os.makedirs(folder)
    except:
        pass
    with open(filename, 'w') as outfile:
        json.dump(result, outfile)