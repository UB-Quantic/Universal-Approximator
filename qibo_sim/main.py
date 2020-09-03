import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--layers", default=4, help="Number of random states.", type=int)
parser.add_argument("--method", default='l-bfgs-b', help="Optimization method", type=str)
parser.add_argument("--save", default=True, help="Save historical data", type=bool)
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
import qibo
qibo.set_device("/CPU:0")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(gpus)
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0)])
    except RuntimeError as e:
        print(e)

from classes import ApproximantNN as aNN
import numpy as np
import classes.aux_functions as aux

def main(layers, method, save):
    x = np.linspace(-1, 1, 31)
    data = np.array(x).reshape((31, 1))

    C = aNN.ApproximantNN(layers, data, aux.tanh)
    m_evs = 2e4
    #cma_options = {'maxfevals': m_evs, 'seed':1112172}
    options = {'maxfun': m_evs}

    '''sgd_options = {"nepochs": 1001,
                               "nmessage": 100,
                                "optimizer": "Adamax",
                               "learning_rate": 0.5}'''
    # scipy_options = {'maxfun':int(1e8)}

    C.run_optimization(method, options, compile=True, save=save)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
