from classes import ApproximantNN as aNN
import numpy as np
import classes.aux_functions as aux
import tensorflow as tf
from time import time
from itertools import product

x = np.linspace(-1, 1, 31)
data = np.array(x).reshape((31, 1))

functs = aux.operators_from_ampl_phase(aux.poly, aux.angulator)

C = aNN.ApproximantNN(1, 10, data, functs)

#print(C.dimension)
#print(C.cost_function(C.params))
# C.paint_representation_1D()
r = C.minimize(method='sgd', options={'disp':True})
print(r)
C.paint_representation_1D()
