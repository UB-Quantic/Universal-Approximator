from qupy import Qubits
import numpy as np
import math
import os
from qupy.operator import *

dtype = getattr(np, os.environ.get('QUPY_DTYPE', 'complex128'))
device = int(os.environ.get('QUPY_GPU', 0))


if device >= 0:
    import cupy
    cupy.cuda.Device(device).use()
    xp = cupy
else:
    xp = np

############# IMPORTANTE: QUPY UTILIZA EL ORDEN DE QUBITS INVERSO



def u3(theta3):
    r = xp.array([[np.cos(theta3[0] * 0.5) * np.exp(1j * theta3[1] * 0.5) * np.exp(1j * theta3[2] * 0.5),
                    -np.sin(theta3[0] * 0.5) * np.exp(1j * theta3[1] * 0.5) * np.exp(-1j * theta3[2] * 0.5)],
                   [np.sin(theta3[0] * 0.5) * np.exp(-1j * theta3[1] * 0.5) * np.exp(1j * theta3[2] * 0.5),
                    np.cos(theta3[0] * 0.5) * np.exp(-1j * theta3[1] * 0.5) * np.exp(-1j * theta3[2] * 0.5)]], dtype=dtype)

    return r #change xp -> math?


class QCircuit(Qubits):
    def __init__(self, qubits):
        Qubits.__init__(self, qubits)
        self.qubits = qubits

    def H(self, target):
        self.gate(H, target=self.qubits - 1 - target)

    def X(self, target):
        self.gate(X, target=self.qubits - 1 - target)

    def Y(self, target):
        self.gate(Y, target=self.qubits - 1 - target)

    def Z(self, target):
        self.gate(Z, target=self.qubits - 1 - target)

    def RX(self, theta, target):
        self.gate(rx(theta), self.qubits - 1 - target)

    def RY(self, theta, target):
        self.gate(ry(theta), self.qubits - 1 - target)

    def RZ(self, theta, target):
        self.gate(rz(theta), self.qubits - 1 - target)

    def U3(self, theta3, target):
        self.gate(rz(theta3), self.qubits - 1 - target)

    def CX(self, target, control):
        self.gate(X, target=self.qubits - 1 - target, control=self.qubits - 1 - control)

    def CY(self, target, control):
        self.gate(Y, target=self.qubits - 1 - target, control=self.qubits - 1 - control)

    def CZ(self, target, control):
        self.gate(Z, target=self.qubits - 1 - target, control=self.qubits - 1 - control)

    def CRX(self, theta, target, control):
        self.gate(rx(theta), self.qubits - 1 - target, control=self.qubits - 1 - control)

    def CRY(self, theta, target, control):
        self.gate(ry(theta), self.qubits - 1 - target, control=self.qubits - 1 - control)

    def CRZ(self, theta, target, control):
        self.gate(rz(theta), self.qubits - 1 - target, control=self.qubits - 1 - control)

    def CU3(self, theta3, target, control):
        self.gate(rz(theta3), self.qubits - 1 - target, control=self.qubits - 1 - control)

    def update_state(self, new_state):
        new_state.reshape((2,)*self.qubits)
        self.state = new_state

    def reset(self):
        self.__init__(self.qubits)

    def measure_probs(self, probabilities, n_samples):
        measurements = np.zeros_like(probabilities).flatten()
        cum_probs = np.cumsum(probabilities)
        p = np.random.rand(n_samples)
        pos = np.searchsorted(cum_probs, p)
        for pos_i in pos:
            measurements[pos_i] += 1
        measurements.reshape(probabilities.shape)
        measurements /= np.sum(measurements)

        return measurements

    def get_all_probabilities(self):
        probabilities = np.abs(self.state) ** 2
        return probabilities

    def measure_all(self, n_samples, return_probabilities = False):
        probabilities = self.get_all_probabilities()
        measurements = self.measure_probs(probabilities, n_samples).reshape(probabilities.shape)

        if return_probabilities:
            return measurements, probabilities

        else:
            return measurements

    def get_some_probabilities(self, measured_qubits):
        probabilities = self.get_all_probabilities()
        l = tuple(self.measure_list(measured_qubits))
        con_probabilities = xp.sum(probabilities, axis=l)

        return con_probabilities


    def measure_some_qubits(self, n_samples, measured_qubits, return_probabilities = False):
        con_probabilities = self.get_some_probabilities(measured_qubits)
        measurements = self.measure_probs(con_probabilities, n_samples).reshape(con_probabilities.shape)

        if return_probabilities == False:
            return measurements

        else:
            return measurements, con_probabilities



class GeneralVariationalAnsatz(Qubits):
    def __init__(self, qubits, layers):
        Qubits.__init__(self, qubits)
        self.layers = layers
        self.parameters = xp.random.rand(2 * layers + 1, qubits, 3)
        self.qubits = self.size

    def reset(self):
        self.state = xp.zeros_like(self.state)
        self.state[(0,) * self.qubits] = 1.

    def entangling_layer_1(self, layer):
        for q in range(self.qubits):
            self.gate(u3(self.parameters[2 * layer, q]), target=self.qubits - 1 - q)

        for q in range(0, self.qubits, 2):
            self.gate(Z, control=self.qubits - 1 - q, target=((self.qubits - q - 2) % self.qubits)) # Coge Z?

    def entangling_layer_2(self, layer):
        for q in range(self.qubits):
            self.gate(u3(self.parameters[2 * layer + 1, q]), target=self.qubits - 1 - q)

        for q in range(1, self.qubits, 2):
            self.gate(Z, control=self.qubits - 1 - q, target=(self.qubits - q - 2) % self.qubits)  # Coge Z?

    def final_layer(self):
        for q in range(self.qubits):
            self.gate(u3(self.parameters[-1, q]), target=self.qubits - 1 - q)

    def run(self):
        self.reset()
        for l in range(self.layers):
            self.entangling_layer_1(l)
            self.entangling_layer_2(l)

        self.final_layer()

    def update_parameters(self, new_parameters):
        if self.parameters.shape != new_parameters.shape:
            raise ValueError("Shape is not appropiate")
        self.parameters = new_parameters


class one_qubit_approximant(QCircuit):
    def __init__(self, layers, domain, f):
        QCircuit.__init__(self, 1)
        self.layers = layers
        self.Domain = domain
        self.Domain_tensor = np.array([np.diag(np.array([x_i, 1, 1])) for x_i in domain])
        self.F = f
        self.params_shape = (layers, 3)
        self.params = np.zeros(self.params_shape)
        #El cálculo grande parámetros * x debe hacerse aquí y sólo una vez



    def layer(self, l, point):
        theta = self.params_x.tensor[l, point] # Se puede hacer con tensorflow, pero desde el principio
        self.RY(0, theta[0])
        self.RY(0, theta[1])
        self.RZ(0, theta[2])


    def run(self):
        y = np.empty_like(self.Domain)
        for i, x_i in enumerate(self.Domain):
            self.reset()
            for l in range(self.layers):
                self.layer(l, i)

            y[i] = self.get_all_probabilities()[1]

        return y


    def chi_square(self):
        y = self.run()
        return np.mean((y - self.F(self.Domain))**2) * 0.5


    def update_parameters(self, new_parameters):
        self.params = new_parameters
        self.params_x = tn.ncon([self.params, self.Domain_tensor], [[-1, 1], [-2, 1, -3]])

    def minimizing_function(self, parameters):
        parameters = parameters.reshape(self.params_shape)
        self.update_parameters(parameters)
        current_value = self.chi_square()
        return current_value

    def get_optimal_parameters(self):
        init_parameters = 2 * (np.random.rand(self.params_shape[0] * self.params_shape[1]) - .5)

        result = minimize(self.minimizing_function,init_parameters)
        return result

