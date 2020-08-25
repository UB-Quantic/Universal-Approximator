import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--layers", default=5, help="Number of random states.", type=int)
parser.add_argument("--method", default='cma', help="Optimization method", type=str)
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

# Hay algún problema en el código. Cosas que no funcionan:
# 1) El cronómetro de cma cuenta menos tiempo del real
# 2) El cma no llega al mismo mínimo, es mucho peor. Por supuesto, los parámetros son totalmente distintos -> Mala codificación

def main(layers, method):
    x = np.linspace(-1, 1, 31)
    data = np.array(x).reshape((31, 1))


    C = aNN.ApproximantNN(1, layers, data, [aux.relu])


    cma_options = {'maxfevals': 1e4, 'seed':1112172}

    '''sgd_options = {"nepochs": 1001,
                               "nmessage": 100,
                               "optimizer": "Adamax",
                               "learning_rate": 0.5}'''
    # scipy_options = {'maxfun':int(1e8)}
    r = C.minimize(method=method, options=cma_options)
    C.paint_representation_1D()

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
