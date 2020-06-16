from qibo_sim.classes import ApproximantNN as aNN
import numpy as np
import qibo_sim.classes.aux_functions as aux
import tensorflow as tf
from time import time
from itertools import product

x = np.linspace(-1, 1, 31)
y = np.linspace(-1, 1, 11)
# data = np.array(list(product(x, y)))
data = np.array(x).reshape((31, 1))

#data = tf.constant(data)
functs = aux.operators_from_ampl_phase(aux.poly, aux.angulator)

C = aNN.ApproximantNN(1, 11, data, [aux.relu])

'''params = [ 5.77922035e-05, -3.06826727e-03, -1.57597920e+00, -1.57462077e+00,
        2.76438643e+00, -1.94780543e-01,  1.62045627e+00, -1.65300718e-02,
        2.42194885e+00,  2.53599005e+00, -1.26264931e+00, -3.17140705e+00,
        1.44656608e+00,  2.43617532e+00,  4.07701930e-01, -1.06772689e-01,
       -2.57305149e+00,  1.88343855e+00,  1.26674147e+00, -6.92750658e-03,
       -2.27110700e-01, -3.59068166e-01, -1.46042402e+00,  1.77769809e+00,
       -1.06128005e+00, -8.11115535e-01,  1.47861263e+00, -5.64043336e-01,
       -2.24479869e-01,  1.98540546e-01, -1.39545443e+00, -1.55371996e+00,
        2.76451854e-01,  8.33551092e-01, -1.01756138e+00,  1.57346336e-01,
        3.78594867e-01, -1.45046113e+00, -6.18417562e-02]
params = np.array(params)
print(len(params))
C.set_parameters(params)'''
#print(C.dimension)
#print(C.cost_function(C.params))
# C.paint_representation_1D()
meth='l-bfgs-b'
sgd_options = {"nepochs": 10001,
                           "nmessage": 1000,
                           "optimizer": "Adamax",
                           "learning_rate": 0.1}
r = C.minimize(method=meth, options=sgd_options)
print(meth)
print(r)
C.paint_representation_1D()
