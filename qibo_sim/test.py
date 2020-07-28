from classes import ApproximantNN as aNN
import numpy as np
import classes.aux_functions as aux

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--layers", default=5, help="Number of random states.", type=int)
parser.add_argument("--method", default='cma', help="Optimization method", type=str)

def main(layers, method):
    x = np.linspace(-1, 1, 31)
    data = np.array(x).reshape((31, 1))


    C = aNN.ApproximantNN(1, layers, data, [aux.relu])


    cma_options = {'maxfevals': 1e4}

    '''sgd_options = {"nepochs": 1001,
                               "nmessage": 100,
                               "optimizer": "Adamax",
                               "learning_rate": 0.5}'''
    # scipy_options = {'maxfun':int(1e8)}
    r = C.minimize(method=method, options=cma_options)
    print(r)
    #C.paint_representation_1D()

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
