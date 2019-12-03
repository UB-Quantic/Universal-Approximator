# Author: David López-Núñez <dln492@gmail.com> 
import single_qubit_control

import sys
import os

import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

sys.path.append("C:\\Program Files (x86)\\Labber\\Script")
import Labber
from Labber import ScriptTools



class UniversalApproximator():

    def __init__(self, n_layers=2, measurement_type="SIMULATION",\
                 single_qubit_control_type="ARBITRARY"):
        
        if n_layers not in [2,3,5]:
            print("Number of layers not supported")
            exit()
        
        if measurement_type != "SIMULATION":
            print("Measurement type not supported")
            exit()
        
        if single_qubit_control_type != "ARBITRARY":
            print("Single Qubit control type not supported")
            exit()
        
        self._n_layers = n_layers

        # set path to executable
        ScriptTools.setExePath('C:\\Program Files (x86)\\Labber\\Program\\Measurement.exe')

        relPath = os.path.dirname( "C:\\Users\\Quantum\\Labber\\Scripts\\Universal Approximator\\simulation\\" ) 

        self._measurement = ScriptTools.MeasurementObject(
                        os.path.join( relPath, "Simulation.hdf5"),
                        os.path.join( relPath, "Simulation_Result.hdf5" ) )
        
        self._sqc = single_qubit_control.SingleQubitControl( self._measurement )
        self._sqc.calibrate()

        self._p = []
        self._x = 0

    def update_param(self, p):
        self._p = p
    
    def update_x(self,x):
        self._x = x

    def _calc_theta(self, p0, p1, x):
        theta = p1 + p0 * x
        return theta
    
    def _create_sequence(self):
        
        self._sqc.reset()
        for param in self._p:
            
            self._sqc.add_y_gate( \
                self._calc_theta( param[0], param[1], self._x ))
            self._sqc.add_z_gate( param[2] )
        self._sqc.finish_sequence()
    
    def run(self):
        self._create_sequence()
        pol = self._measurement.performMeasurement()
        P_e = (1 + pol[0])/2
        return P_e
    
if __name__ == "__main__":
    pass

            



