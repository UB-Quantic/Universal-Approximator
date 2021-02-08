import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--method", default='cma', help="Optimization method", type=str)
parser.add_argument("--real", default='relu', help="Function to fit", type=str)
parser.add_argument("--imag", default='poly', help="Function to fit", type=str)
parser.add_argument("--ansatz", default='Weighted', help="Ansatz ", type=str)

import numpy as np
from ApproximantNN import Approximant_complex as App_c
from importlib import import_module

def main(real, imag, method, ansatz):
    real = globals()[f"{real}"]

    x = np.linspace(-1, 1, 31)
    #data = np.array(x).reshape((31, 1))
    imag = globals()[f"{imag}"]
    for layers in range(6, 7):
        for seed in range(11):
            C = App_c(layers, x, ansatz, real, imag)
            #C.run_optimization(method, options={'maxiter':10000}, compile=True, seed=seed)
            C.run_optimization_classical('bfgs', options={'maxiter': 10000}, seed=seed)
#C.paint_representation_1D('lbfgsb_%s.pdf'%layers)


def step(x):
    step.name = 'step'
    return 0.5 * (np.sign(x) + 1)

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
