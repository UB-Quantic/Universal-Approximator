import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--layers", default=4, help="Number of random states.", type=int)
parser.add_argument("--method", default='powell', help="Optimization method", type=str)
parser.add_argument("--save", default=False, help="Save historical data", type=bool)
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

def main(layers, method, save):
    np.random.seed(5)
    x = np.linspace(-1, 1, 31)
    data = np.array(x).reshape((31, 1))

    C = aNN.ApproximantNN(layers, data, aux.relu)
    m_evs = 2e4
    #cma_options = {'maxfevals': m_evs, 'seed':1112172}
    options = {'maxfev': m_evs}

    '''sgd_options = {"nepochs": 1001,
                               "nmessage": 100,
                                "optimizer": "Adamax",
                               "learning_rate": 0.5}'''
    # scipy_options = {'maxfun':int(1e8)}
    r = C.minimize(method=method, options=options, save_historical=save)
    print(r)
    name = method + '_%s.pdf'%(layers)
    C.paint_representation_1D(name)
    outcomes = np.zeros_like(C.domain)
    for j, x in enumerate(C.domain):
        state = C.get_state(x)
        outcomes[j] = C.hamiltonian[0].expectation(state)

    print((-outcomes.flatten() + 1) / 2)
    # C.paint_representation_1D(name)
    if save:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=2)
        axs[0].plot(np.arange(len(C.hist_chi)), C.hist_chi)
        axs[0].set(yscale='log', ylabel=r'$\chi^2$')
        hist_params = np.array(C.hist_params)
        for i in range(len(C.params)):
            axs[1].plot(np.arange(len(C.hist_chi)), hist_params[:, i], 'C%s'%i)

        axs[1].set(ylabel='Parameter', xlabel='Function evaluation')
        fig.suptitle('Historical behaviour', fontsize=16)
        fig.savefig(method + '_%s_historical.pdf'%(layers))



if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
