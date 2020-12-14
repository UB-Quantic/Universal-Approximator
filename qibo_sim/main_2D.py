import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--method", default='l-bfgs-b', help="Optimization method", type=str)
parser.add_argument("--function", default='threehump', help="Function to fit", type=str)
parser.add_argument("--ansatz", default='Weighted_2D', help="Ansatz", type=str)

import numpy as np
from classes.aux_functions import *
from classes.ApproximantNN import Approximant_real_2D as App_r
from importlib import import_module

def main(function, method, ansatz):
    x_0 = np.linspace(-5, 5, 25)
    x_1 = np.linspace(-5, 5, 25)
    from itertools import product
    x = np.array(list(product(x_0, x_1)))
    for layers in range(2, 3):
        for seed in range(1):
            func = globals()[f"{function}"]

            C = App_r(layers, x, func, ansatz)
            C.run_optimization(method, options={'maxiter':1000}, compile=True, seed=seed)
            C.run_optimization_classical('l-bfgs-b', options={'maxiter': 10000, 'disp':True}, seed=seed)
#C.paint_representation_1D('lbfgsb_%s.pdf'%layers)'''

def himmelblau(x):
    himmelblau.name = 'himmelblau'
    for x_ in x:
        yield (x_[0]**2 + x_[1] - 11)**2 + (x_[0] + x_[1]**2 - 7)**2

def brent(x):
    brent.name = 'brent'
    for x_ in x:
        yield np.linalg.norm(x_ / 2)**2 + np.exp(-np.linalg.norm(x_/2 - 5)**2)

def threehump(x):
    threehump.name = 'threehump'
    for x_ in x * 2 / 5:
        yield 2 * x_[0]**2 - 1.05 * x_[0]**4 + 1/6 * x_[0]**6 + x_[0] * x_[1] + x_[1]**2

def adjiman(x):
    adjiman.name = 'adjiman'
    for x_ in x:
        yield np.cos(x_[0]) * np.sin(x_[1]) - x_[0] / (x_[1]**2 + 1)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
