import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--method", default='cma', help="Optimization method", type=str)
parser.add_argument("--function", default='poly', help="Function to fit", type=str)
parser.add_argument("--ansatz", default='Weighted', help="Ansatz ", type=str)

import numpy as np
from ApproximantNN import Approximant_real as App_r
from importlib import import_module

def main(function, method, ansatz):
    for layers in range(4,5):
        for seed in range(5,11):
            func = globals()[f"{function}"]

            x = np.linspace(-1, 1, 31)
            #data = np.array(x).reshape((31, 1))

            C = App_r(layers, x, ansatz, func)
            #C.run_optimization(method, options={'maxiter':10000}, compile=True, seed=seed)
            C.run_optimization_classical('l-bfgs-b', options={'maxiter': 10000}, seed=seed)


def step(x):
    step.name = 'step'
    return 0.5 * (np.sign(x) + 1)


def sigmoid(x, a=10):
    sigmoid.name = 'sigmoid'
    return 1 / (1 + np.exp(-a * x))

def tanh(x, a=5):
    tanh.name = 'tanh'
    return np.tanh(a * x)

def relu(x):
    relu.name = 'relu'
    return x * (x > 0)

def poly(x):
    poly.name= 'poly'
    return np.abs(3*x**3 * (1 - x**4))

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
