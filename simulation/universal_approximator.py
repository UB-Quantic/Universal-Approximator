# Author: David López-Núñez <dln492@gmail.com> 
import os
from abc import ABC, abstractmethod
import numpy as np

from Labber import ScriptTools

import single_qubit_control as sqc

MEAS_TYPE_SIMULATION = "SIMULATION"
MEAS_TYPE_EXPERIMENT = "EXPERIMENT"


SIM_MEAS_TEMPLATE = "Simulation.hdf5"
SIM_MEAS_RESULT = "Simulation_Result.hdf5"
EXP_MEAS_TEMPLATE = "Experiment.hdf5"
EXP_MEAS_RESULT = "Experiment_Result.hdf5"

_RELPATH = os.path.dirname(os.path.abspath(__file__))


##### add single qubit control constants ####


class UniversalApproximator():

    def __init__(self, n_layers=2, measurement_type="SIMULATION",\
                 single_qubit_control_type="ARBITRARY"):
        
        if n_layers not in [2,3,5]:
            print("Number of layers not supported")
            exit()
        
        if single_qubit_control_type != "ARBITRARY":
            print("Single Qubit control type not supported")
            exit()
        
        self._executer = _UnivApproxFactory.get_executer( n_layers, measurement_type )

        self._n_layers = n_layers


    def update_param(self, p):

        if len(p) != self._n_layers or len(p[0]) != 3:
            print("Parameters don't have the proper length")
            exit()

        self._executer.update_param(p)
    
    def update_x(self,x):

        self._executer.update_x(x)

    
    def run(self):
        P0 = self._executer.run()
        return P0


class _UnivApproxFactory():

    @staticmethod
    def get_executer( n_layers, measurement_type, ):
        if measurement_type == MEAS_TYPE_EXPERIMENT:
            return _UnivApproxExperiment(n_layers, "ARBITRARY")
        elif measurement_type == MEAS_TYPE_SIMULATION:
            return _UnivApproxSimulation(n_layers, "ARBITRARY")
        

class _UnivApproxExecuter(ABC):

    def __init__(self, n_layers, single_qubit_control_type):
              
        self._n_layers = n_layers

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

    @abstractmethod
    def _convert_to_P0(self, result):
        return result        
        
    def run(self):
        self._create_sequence()
        result = self._measurement.performMeasurement()
        P0 = self._convert_to_P0(result)
        return P0


class _UnivApproxSimulation( _UnivApproxExecuter ):

    def __init__(self, n_layers, single_qubit_control_type):
        
        self._measurement = ScriptTools.MeasurementObject(
                        os.path.join( _RELPATH, SIM_MEAS_TEMPLATE),
                        os.path.join( _RELPATH, SIM_MEAS_RESULT ) )
        self._sqc = sqc.SQCFactory.get_single_qubit_controller( \
             self._measurement, meas_type = MEAS_TYPE_SIMULATION)

        _UnivApproxExecuter.__init__(self, n_layers, single_qubit_control_type)


    def _convert_to_P0(self, result):
        
        P0 = (1 + result[0])/2 
        return P0



class _UnivApproxExperiment( _UnivApproxExecuter ):

    def __init__(self, n_layers, single_qubit_control_type):
        
        self._measurement = ScriptTools.MeasurementObject(
                        os.path.join( _RELPATH, EXP_MEAS_TEMPLATE),
                        os.path.join( _RELPATH, EXP_MEAS_RESULT ) )
        self._sqc = sqc.SQCFactory.get_single_qubit_controller( \
             self._measurement, meas_type = MEAS_TYPE_EXPERIMENT)

        _UnivApproxExecuter.__init__(self, n_layers, single_qubit_control_type)

    def _convert_to_P0(self, result):
        
        P0 = self._sqc.process_result(np.abs(result[1]))
        return P0


if __name__ == "__main__":
    pass

            



