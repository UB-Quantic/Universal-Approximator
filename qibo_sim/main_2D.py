import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--method", default='cma', help="Optimization method", type=str)
parser.add_argument("--function", default='tanh_2D', help="Function to fit", type=str)
parser.add_argument("--ansatz", default='Fourier_2D', help="Ansatz", type=str)

import numpy as np
from classes.aux_functions import *
from classes.ApproximantNN import Approximant_real_2D as App_r
from importlib import import_module

def main(function, method, ansatz):
    x_0 = np.linspace(-1, 1, 15)
    x_1 = np.linspace(-1, 1, 15)
    from itertools import product
    x = np.array(list(product(x_0, x_1)))
    for layers in range(5,6):
        for seed in range(1):
            func = globals()[f"{function}"]

            C = App_r(layers, x, func, ansatz)
            C.run_optimization(method, options={'maxiter':100}, compile=True, seed=seed)
            C.run_optimization_classical('l-bfgs-b', options={'maxiter': 10000, 'disp':True}, seed=seed)
#C.paint_representation_1D('lbfgsb_%s.pdf'%layers)'''

def tanh_2D(x, a=5):
    tanh_2D.name = 'tanh_2D'
    for x_ in x:
        yield np.tanh(a * (np.linalg.norm(x_) - 1/ 2))

def paraboloid(x):
    paraboloid.name = 'paraboloid'
    for x_ in x:
        yield np.linalg.norm(x_)**2

def relu_2D(x):
    relu_2D.name = 'relu_2D'
    for x_ in x:
        yield max(0, x_[1] - x_[0])

def hyperboloid(x):
    hyperboloid.name = 'hyperboloid'
    for x_ in x:
        yield x_[1]**2 - x_[0]**2


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
