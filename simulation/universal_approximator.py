# Author: David López-Núñez <dln492@gmail.com> 
import os
from abc import ABC, abstractmethod
import numpy as np

from Labber import ScriptTools

import single_qubit_control as sqc

# Constants that will be used throughout the code
MEAS_TYPE_SIMULATION = "SIMULATION"
MEAS_TYPE_EXPERIMENT = "EXPERIMENT"

_RELPATH = os.path.dirname(os.path.abspath(__file__))


class UniversalApproximator():
    """
    Class that uses the qubit as a Universal Approximator.
    
    It can be used both in simulation and in real experiment.
    """

    def __init__(self, n_layers=2, measurement_type="SIMULATION",\
                 pulse_type="GAUSS_PLAT", features= {}):
        """
        Initialize the class by sending the measurement type and the number of layers
        of the algorithm. If known, also the type of qubit control is sent
        """
        
        # Check if values are correct and some initializations
        if n_layers not in range(1,6):
            print("Number of layers not supported")
            exit()
        self._n_layers = n_layers
        
        if measurement_type not in [MEAS_TYPE_EXPERIMENT, MEAS_TYPE_SIMULATION]:
            print("Measurement types not supported")
            exit()
        self._meas_type = measurement_type
        
        if pulse_type not in ["GAUSS_PLAT","GAUSSIAN"]:
            print("Single Qubit control type not supported")
            exit()
        
        if measurement_type == MEAS_TYPE_EXPERIMENT:
            meas_file = "Exp_"
        elif measurement_type == MEAS_TYPE_SIMULATION:
            meas_file = "Sim_"

        if pulse_type == "GAUSS_PLAT":
            meas_file += "Gpl"
            cal_file = meas_file + "_Cal"
        elif pulse_type == "GAUSSIAN":
            meas_file += "Gau"
            cal_file = meas_file + "_Cal"
        
        meas_result = meas_file + "_Res.hdf5"
        meas_file +=  ".hdf5"
        cal_file +=  ".hdf5"

        self._p = []
        self._x = 0
        self.theta = []

        self._features = features

        self._measurement = ScriptTools.MeasurementObject(
                        os.path.join( _RELPATH, meas_file),
                        os.path.join( _RELPATH, meas_result ) )

        # Creates and calibrates the single qubit controller
        self._sqc = sqc.SingleQubitControl(self._measurement, \
            pulse_type, cal_file, features)

        self._sqc.calibrate()        

    def update_param(self, p):
        """
        Update the parameters of the algorithm. This should be a 
        (n_layers x 3) array
        """

        # if len(p) != self._n_layers or len(p[0]) != 3:
        #     print(len(p))
        #     print(len(p[0]))
        #     print("Parameters don't have the proper length")
        #     exit()

        p_i = []
        p_i.append(p)
        self._p = p_i
    
    def update_x(self,x):
        """
        Update the value of x to be evaluated
        """

        self._x = x

    def set_x_range(self,x):
        """
        Set x range to be computed
        """

        self._x_range = x

    def _calc_theta(self, p0, p1, x):
        """
        Provide the parameters and the x value, it calculates the
        corresponding theta: theta= p1 + p0 * x
        """
        theta = p1 + p0 * x
        return theta
    
    def _create_sequence(self):
        """
        Creates the sequence once the parameters and the x value has been
        supplied
        """
        # First a reset is done to previous pulses
        self._sqc.reset()
        self.theta = []

        if self._features["reset"] == True:
            self._sqc.add_pre_reset()
        # For each layer, the following pulse are added:
        # Ry(p1 + p0*x) -> Rz(p2)
        for param in self._p:
            theta = self._calc_theta( param[0], param[1], self._x )
            self.theta.append( theta )
            self._sqc.add_y_gate( theta )
            self._sqc.add_z_gate( param[2] )
        
        self._sqc.finish_sequence()

    def _convert_to_P0(self, result):
        """
        Method to be overriden from subclasses
        """
        P0 = self._sqc.process_result(np.abs(result[1]))
        return P0
        
    def run(self):
        """
        Run the experiment
        """

        self._create_sequence()
        result = self._measurement.performMeasurement()
        P0 = self._convert_to_P0(result)
        return P0

    def probability(self, p, x):
        self.update_param(p)
        P_1 = [] 
        for x_i in self._x_range:
            self.update_x(x_i)
            P_0_value = self.run()
            # Result is P0 and we want P1. We add it to the array.
            P_1.append( 1 - P_0_value )
        return P_1

    def chi_square(self, p):
        P_1 = self.probability(p, self._x_range)
        fx = self._function(self._x_range)
        result = ( 0.5 / len(self._x_range) ) * np.sum( ( P_1 - fx )**2 )
        return result
        
    def define_function(self,f):
        self._function = f

if __name__ == "__main__":
    pass

            



