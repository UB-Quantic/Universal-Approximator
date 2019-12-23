# Author: David López-Núñez <dln492@gmail.com> 
import os
from abc import ABC, abstractmethod
import numpy as np

from Labber import ScriptTools

import single_qubit_control as sqc

# Constants that will be used throughout the code
MEAS_TYPE_SIMULATION = "SIMULATION"
MEAS_TYPE_EXPERIMENT = "EXPERIMENT"

# SIM_MEAS_TEMPLATE = "Simulation.hdf5"
SIM_MEAS_TEMPLATE = "Simulation_No_noise.hdf5"
SIM_MEAS_RESULT = "Simulation_Result.hdf5"
EXP_MEAS_TEMPLATE = "Experiment_No_spa.hdf5"
EXP_MEAS_RESULT = "Experiment_Result.hdf5"

_RELPATH = os.path.dirname(os.path.abspath(__file__))


class UniversalApproximator():
    """
    Class that uses the qubit as a Universal Approximator.
    
    It can be used both in simulation and in real experiment.
    """

    def __init__(self, n_layers=2, measurement_type="SIMULATION",\
                 single_qubit_control_type="ARBITRARY"):
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
        
        if single_qubit_control_type != "ARBITRARY":
            print("Single Qubit control type not supported")
            exit()

        # get the actual executer of the algorithm        
        self._executer = _UnivApproxFactory.get_executer( n_layers, measurement_type )


    def update_param(self, p):
        """
        Update the parameters of the algorithm. This should be a 
        (n_layers x 3) array
        """

        if len(p) != self._n_layers or len(p[0]) != 3:
            print("Parameters don't have the proper length")
            exit()

        self._executer.update_param(p)
    
    def update_x(self,x):
        """
        Update the value of x to be evaluated
        """

        self._executer.update_x(x)

    
    def run(self):
        """
        Run the experiment
        """

        P0 = self._executer.run()
        return P0


class _UnivApproxFactory():
    """
    Factory class for creating the Universal Approximator algorithm
    executor depending on if it's a simulation or a experiment
    """

    @staticmethod
    def get_executer( n_layers, measurement_type, ):
        """
        Returns the suitable executer depending on the measurement type
        ( "SIMULATION" or "EXPERIMENT"). Number of layers must also be supplied.
        """
        if measurement_type == MEAS_TYPE_EXPERIMENT:
            return _UnivApproxExperiment("ARBITRARY")
        elif measurement_type == MEAS_TYPE_SIMULATION:
            return _UnivApproxSimulation("ARBITRARY")
        

class _UnivApproxExecuter(ABC):
    """
    Abstract class that implements most of the features
    of the universal approximator executer
    """

    def __init__(self, single_qubit_control_type):
        """
        Initialize the basic parameters of the Universal Approximator Executer
        """
        self._p = []
        self._x = 0

    def update_param(self, p):
        """
        Update the algorithm parameters
        """
        self._p = p
    
    def update_x(self,x):
        """
        Update the x value to be evaluated
        """
        self._x = x

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

        # For each layer, the following pulse are added:
        # Ry(p1 + p0*x) -> Rz(p2)
        for param in self._p:
            
            self._sqc.add_y_gate( \
                self._calc_theta( param[0], param[1], self._x ))
            self._sqc.add_z_gate( param[2] )
        
        self._sqc.finish_sequence()

    @abstractmethod
    def _convert_to_P0(self, result):
        """
        Method to be overriden from subclasses
        """
        return result        
        
    def run(self):
        """
        Run the experiment
        """

        self._create_sequence()
        result = self._measurement.performMeasurement()
        P0 = self._convert_to_P0(result)
        return P0


class _UnivApproxSimulation( _UnivApproxExecuter ):
    """
    Executer class of the universal approximator algorithm
    when in simulation mode
    """

    def __init__(self, single_qubit_control_type):
        
        # Creates the Labber measurement object
        self._measurement = ScriptTools.MeasurementObject(
                        os.path.join( _RELPATH, SIM_MEAS_TEMPLATE),
                        os.path.join( _RELPATH, SIM_MEAS_RESULT ) )
        
        # Creates and calibrates the single qubit controller
        self._sqc = sqc.SQCFactory.get_single_qubit_controller( \
             self._measurement, meas_type = MEAS_TYPE_SIMULATION)
        self._sqc.calibrate()

        # Performs superclass initialization
        _UnivApproxExecuter.__init__(self, single_qubit_control_type)


    def _convert_to_P0(self, result):
        """
        Easily convert polarization to P0
        """
        P0 = (1 + result[0])/2 
        return P0



class _UnivApproxExperiment( _UnivApproxExecuter ):
    """
    Executer class of the universal approximator algorithm
    when in experiment mode
    """

    def __init__(self, single_qubit_control_type):
        
        # Creates the Labber measurement object
        self._measurement = ScriptTools.MeasurementObject(
                        os.path.join( _RELPATH, EXP_MEAS_TEMPLATE),
                        os.path.join( _RELPATH, EXP_MEAS_RESULT ) )

        # Creates and calibrates the single qubit controller
        self._sqc = sqc.SQCFactory.get_single_qubit_controller( \
             self._measurement, meas_type = MEAS_TYPE_EXPERIMENT)
        self._sqc.calibrate()


        # Performs superclass initialization
        _UnivApproxExecuter.__init__(self, single_qubit_control_type)

    def _convert_to_P0(self, result):
        """
        Uses the single qubit control class to process the result
        """
        P0 = self._sqc.process_result(np.abs(result[1]))
        return P0


if __name__ == "__main__":
    pass

            



