import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--method", default='cma', help="Optimization method", type=str)
parser.add_argument("--function", default='poly', help="Function to fit", type=str)
parser.add_argument("--ansatz", default='Weighted', help="Ansatz ", type=str)

import numpy as np
from classes.aux_functions import *
from classes.ApproximantNN import Approximant_real as App_r
from importlib import import_module

def main(function, method, ansatz):
    for layers in range(1, 7):
        for seed in range(3):
            func = globals()[f"{function}"]

            x = np.linspace(-1, 1, 31)
            #data = np.array(x).reshape((31, 1))

            C = App_r(layers, x, ansatz, func)
            C.run_optimization(method, options={'maxiter':10000}, compile=True, seed=seed)
            C.run_optimization_classical('l-bfgs-b', options={'maxiter': 10000}, seed=seed)
#C.paint_representation_1D('lbfgsb_%s.pdf'%layers)


'''params = np.array([0.7899151070137848, 1.441640364909414, -0.2819381075486134,
 -1.0591725918927786, -3.0108313911015605, 1.8243932523358068,
 0.5293709400517546, -3.135372294754921, 1.0242983398335992,
 -0.5293922230189926, 3.149793936494836, 376.88339132235194]
)
C.set_parameters(params)
C.paint_representation_1D('test.pdf')'''
'''def main(layers, method, save):
    x = np.linspace(-1, 1, 31)
    data = np.array(x).reshape((31, 1))

    C = aNN.ApproximantNN(layers, data, aux.tanh)
    m_evs = 2e4
    #cma_options = {'maxfevals': m_evs, 'seed':1112172}
    options = {'maxfun': m_evs}

    # scipy_options = {'maxfun':int(1e8)}

    C.run_optimization(method, options, compile=True, save=save)'''

def step(x):
    step.name = 'step'
    return 0.5 * (np.sign(x) + 1)

def cosine(x):
    cosine.name = 'cosine'
    return np.cos(2*np.pi*x)

def sigmoid(x, a=10):
    sigmoid.name = 'sigmoid'
    return 1 / (1 + np.exp(-a * x))

def tanh(x, a=5):
    tanh.name = 'tanh'
    return np.tanh(a * x)


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
    return x * (x > 0)

def poly(x):
    poly.name= 'poly'
    return np.abs(3*x**3 * (1 - x**4))

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
