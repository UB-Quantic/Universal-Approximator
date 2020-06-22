from classes import ApproximantNN as aNN
import numpy as np
import classes.aux_functions as aux

x = np.linspace(-1, 1, 31)
data = np.array(x).reshape((31, 1))


C = aNN.ApproximantNN(1, 5, data, [aux.relu])

meth='l-bfgs-b'
'''sgd_options = {"nepochs": 1001,
                           "nmessage": 100,
                           "optimizer": "Adamax",
                           "learning_rate": 0.5}'''
scipy_options = {'maxfun':int(1e10)}
r = C.minimize(method=meth, options=scipy_options)
print(r)
C.paint_representation_1D()
