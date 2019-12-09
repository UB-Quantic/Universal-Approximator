# Author: David López-Núñez <dln492@gmail.com> 
import os

from Labber import ScriptTools

import single_qubit_control

_MEAS_TYPE_SIMULATION = "SIMULATION"
_MEAS_TYPE_EXPERIMENT = "EXPERIMENT"


_SIM_MEAS_TEMPLATE = "Simulation.hdf5"
_SIM_MEAS_RESULT = "Simulation_Result.hdf5"
_EXP_MEAS_TEMPLATE = "Experiment.hdf5"
_EXP_MEAS_RESULT = "Experiment_Result.hdf5"


##### add single qubit control constants


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


        relPath = os.path.dirname(os.path.abspath(__file__))
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
        P0 = (1 + pol[0])/2
        return P0


class _UnivApproxFactory():

    @staticmethod
    def get_executer( measurement_type ):
        if measurement_type == _MEAS_TYPE_EXPERIMENT:
            return _UnivApproxExperiment()
        elif measurement_type == _MEAS_TYPE_SIMULATION:
            return _UnivApproxSimulation()
        

class _UnivApproxExecuter():
    # def __init__(self, n_layers=2, measurement_type="SIMULATION",\
    #              single_qubit_control_type="ARBITRARY"):
        
    #     if n_layers not in [2,3,5]:
    #         print("Number of layers not supported")
    #         exit()
        
    #     if measurement_type != "SIMULATION":
    #         print("Measurement type not supported")
    #         exit()
        
    #     if single_qubit_control_type != "ARBITRARY":
    #         print("Single Qubit control type not supported")
    #         exit()
        
    #     self._n_layers = n_layers


    #     relPath = os.path.dirname(os.path.abspath(__file__))
    #     self._measurement = ScriptTools.MeasurementObject(
    #                     os.path.join( relPath, "Simulation.hdf5"),
    #                     os.path.join( relPath, "Simulation_Result.hdf5" ) )
        
    #     self._sqc = single_qubit_control.SingleQubitControl( self._measurement )
    #     self._sqc.calibrate()

    #     self._p = []
    #     self._x = 0

    # def update_param(self, p):
    #     self._p = p
    
    # def update_x(self,x):
    #     self._x = x

    # def _calc_theta(self, p0, p1, x):
    #     theta = p1 + p0 * x
    #     return theta
    
    # def _create_sequence(self):
        
    #     self._sqc.reset()
    #     for param in self._p:
            
    #         self._sqc.add_y_gate( \
    #             self._calc_theta( param[0], param[1], self._x ))
    #         self._sqc.add_z_gate( param[2] )
    #     self._sqc.finish_sequence()
    
    # def run(self):
    #     self._create_sequence()
    #     pol = self._measurement.performMeasurement()
    #     P0 = (1 + pol[0])/2
    #     return P0


class _UnivApproxSimulation( _UnivApproxExecuter ):
    def __init__(self):
        pass

class _UnivApproxExperiment( _UnivApproxExecuter ):
    def __init__(self):
        pass

if __name__ == "__main__":
    pass

            



