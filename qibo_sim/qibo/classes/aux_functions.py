import numpy as np
import os
import json
from qibo.hamiltonians import Hamiltonian
from qibo.config import matrices

def operators_from_ampl_phase(f_ampl, f_phase):
    z = lambda *a, **kw: 1 - 2*f_ampl(*a, **kw)
    x = lambda *a, **kw: 2 * np.sqrt(f_ampl(*a, **kw)) * np.sqrt(1 - f_ampl(*a, **kw)) * np.cos(f_phase(*a, **kw))
    y = lambda *a, **kw: 2 * np.sqrt(f_ampl(*a, **kw)) * np.sqrt(1 - f_ampl(*a, **kw)) * np.sin(f_phase(*a, **kw))
    return [z, x, y]

def operators_from_ampl_phase_entangled(f_ampl, f_phase):
    z = lambda *a, **kw: 1 - 2*f_ampl(*a, **kw)
    x = lambda *a, **kw: 2 * np.sqrt(f_ampl(*a, **kw)) * np.sqrt(1 - f_ampl(*a, **kw)) * np.cos(f_phase(*a, **kw))
    y = lambda *a, **kw: np.zeros_like(*a, **kw)
    return [z, x, y]

def step(x):
    step.name = 'step'
    return 0.5 * (np.sign(x) + 1)

def cosine(x):
    cosine.name = 'cosine'
    return 0.5 * (np.cos(2*np.pi*x) + 1)

def sigmoid(x):
    sigmoid.name = 'sigmoid'
    return 1 / (1 + np.exp(-10 * x))

def tanh(x):
    tanh.name = 'tanh'
    #return 0.5 * (np.tanh(x) + 1)
    return np.tanh(x)

def angulator(x):
    return 2 * np.pi * x

def zero(x):
    tanh.name = 'zero'
    #return 0.5 * (np.tanh(x) + 1)
    return 0

def tanh_2(x):
    tanh.name = 'tanh'
    #return 0.5 * (np.tanh(x) + 1)
    return (np.tanh(np.linalg.norm(x))**2)

def relu(x):
    relu.name = 'relu'
    return np.clip(x, 0, np.max(x))

def poly(x):
    poly.name= 'poly'
    return np.abs(3*x**3 * (1 - x**4))

def minimize_with_previous_result(algorithm, layers, domain, f, phase=None, gens=100, tol=1e-8):
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
    print(res_q)

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

def hamiltonian(measurements):
    nqubits = len(measurements[0])
    H = [Hamiltonian(nqubits)]*len(measurements)
    for n, measur in enumerate(measurements):
        measur = _label_to_matrix(measur)
        h_ = np.kron(measur[-2], measur[-1])
        for m in measur[-3::-1]:
            h_ = np.kron(m, h_)

        print(h_)
        H[n] = h_

    return H

def _label_to_hamiltonian(labels):
    mats = [[]]*len(labels)
    for j, label in enumerate(labels):
        if label == 'I':
            mats[j] = matrices._npI()
        if label == 'X':
            mats[j] = matrices._npX()
        if label == 'Y':
            mats[j] = matrices._npY()
        if label == 'Z':
            mats[j] = matrices._npZ()

    return mats

