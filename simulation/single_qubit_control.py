# Author: David López-Núñez <dln492@gmail.com> 
import os
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from Labber import ScriptTools

MEAS_TYPE_SIMULATION = "SIMULATION"
MEAS_TYPE_EXPERIMENT = "EXPERIMENT"


SIM_CAL_TEMPLATE = "Sim_Calibration.hdf5"
SIM_CAL_RESULT = "Sim_Calibration_Result.hdf5"
EXP_CAL_TEMPLATE = "Exp_Calibration.hdf5"
EXP_CAL_RESULT = "Exp_Calibration_Result.hdf5"

def general_cosinus(t, w, phi, A, c):
    return A * np.cos(w * t + phi) + c

class SingleQubitControl(ABC):
    """
    A class that controls the qubit
    """
    def __init__(self, meas_object, pulse_type = "ARBITRARY"):
        
        self._meas_object = meas_object
        self._pulse_type = pulse_type

        # Prepares calibration measurement to be done
        relPath = os.path.dirname(os.path.abspath(__file__))
        self._calibration_meas = ScriptTools.MeasurementObject(
                os.path.join( relPath, self._calibration_file),
                os.path.join( relPath, self._calibration_file_result ) )

        # Basic initialization
        self._w, self._phi_0, self._A, self._c = 0,0,0,0
        self._accumulated_z_phase, self._next_pulse = 0,1 # initialize variables

    
    def calibrate(self):
        """
        Calibrate both the qubit pulses and the qubit measurement
        """
        # Perform calibration measurement
        ( t_cal, y_cal ) = self._calibration_meas.performMeasurement()
        
        # Adjust oscillations to cosinus and retrieve values
        self._w, self._phi_0, self._A, self._c = self._fit_cosinus( t_cal, y_cal )
    
    def process_result(self, result):
        """
        Process result and convert it into P_0
        """
        P0 = ( result - (self._c - self._A) ) / (2 * self._A)
        return P0

    
    def _fit_cosinus(self, t_cal, y_cal):
        """
        Fit the calibration result to a general cosinus
        """
        # First some trials are obtained so that the fit is succesful
        max_y = np.amax(y_cal)
        min_y = np.amin(y_cal)
        A_trial = abs(max_y - min_y) / 2
        c_trial = (max_y + min_y) / 2

        # w and phi_0 are kind of known in advance
        param_ini = [25e6 * 2*np.pi, 0.1, A_trial, c_trial]
        parameters, par_errors = curve_fit( general_cosinus, t_cal, y_cal, param_ini)

        # mapping parameters
        w = parameters[0]
        phi = parameters[1]
        A = parameters[2]
        c = parameters[3]

        # Plot results in case you want to be sure of the calibration
        # plt.figure(1)
        # plt.plot(t_cal,y_cal)
        # y_sim = []
        # y_trial = []
        # for t in t_cal:
        #     y_sim.append(\
        #         general_cosinus(t, w, phi, A, c))
        #     y_trial.append(\
        #         general_cosinus(t, 25e6 * 2*np.pi, 0.1, 1, 0))

        # plt.plot(t_cal, y_sim)
        # plt.plot(t_cal, y_trial)
        # plt.show()
        return w, phi, A, c


    def _get_time_for_angle(self, angle):
        """
        Obtain the pulse time for a desired angle rotation
        """
        
        if angle < self._phi_0:
            angle = angle + 2*np.pi
        pulse_time = (angle - self._phi_0) / self._w

        return pulse_time


    def add_x_gate(self, angle):
        """
        Add an x gate with specific angle to the sequence
        """
        pulse_plat_name = "Control Pulse - Plateau #" + str(self._next_pulse)
        phase_plat_name = "Control Pulse - Phase #" + str(self._next_pulse)

        self._meas_object.updateValue( pulse_plat_name, self._get_time_for_angle(angle) )
        self._meas_object.updateValue( phase_plat_name, self._accumulated_z_phase )
    
        self._next_pulse += 1

    
    def add_y_gate(self, angle):
        """
        Add an y gate with specific angle to the sequence
        """
        pulse_plat_name = "Control Pulse - Plateau #" + str(self._next_pulse)
        phase_plat_name = "Control Pulse - Phase #" + str(self._next_pulse)

        self._meas_object.updateValue( pulse_plat_name, self._get_time_for_angle(angle) )
        self._meas_object.updateValue( phase_plat_name, self._accumulated_z_phase + 90 )
    
        self._next_pulse += 1


    def add_z_gate(self, angle):
        """
        Add an z gate with specific angle to the sequence
        """
        # Z gates are actually virtual Z gates, so they are only accumulated phase
        # in next x-y pulses
        self._accumulated_z_phase += np.rad2deg( angle )

    
    def finish_sequence(self):
        """
        Finish sequence so that it's ready for measurement
        """
        self._meas_object.updateValue( "Control Pulse - # of pulses", self._next_pulse - 1 )


    def reset(self):
        """
        Reset parameters when a new sequence is started
        """
        self._meas_object.updateValue( "Control Pulse - # of pulses", 0 )
        self._next_pulse = 1
        


class SQCSim(SingleQubitControl):
    """
    A class that controls the qubit when in simulation mode
    """
    def __init__(self, meas_object, pulse_type = "ARBITRARY"):

        # Initialize the single qubit control object with proper calibration
        # files
        self._calibration_file = SIM_CAL_TEMPLATE
        self._calibration_file_result = SIM_CAL_RESULT
        SingleQubitControl.__init__(self, meas_object, pulse_type)        



class SQCExp(SingleQubitControl):
    """
    A class that controls the qubit when in experiment mode
    """
    def __init__(self, meas_object, pulse_type = "ARBITRARY"):
        
        self._calibration_file = EXP_CAL_TEMPLATE
        self._calibration_file_result = EXP_CAL_RESULT
        
        SingleQubitControl.__init__(self, meas_object, pulse_type)        
    

    def add_x_gate(self, angle):
        """
        Add an x gate with specific angle to the sequence
        """

        pulse_amp_name = "Control Pulse - Amplitude #" + str(self._next_pulse)
        pulse_plat_name = "Control Pulse - Plateau #" + str(self._next_pulse)
        pulse_phase_name = "Control Pulse - Phase #" + str(self._next_pulse)
        pulse_output_name = "Control Pulse - Output #" + str(self._next_pulse)

        plateau = self._get_time_for_angle(angle)

        self._meas_object.updateValue( pulse_amp_name, 1 )
        self._meas_object.updateValue( pulse_plat_name, plateau )
        self._meas_object.updateValue( pulse_phase_name, self._accumulated_z_phase )
        self._meas_object.updateValue( pulse_output_name, 0 )
    
        self._next_pulse += 1
        self._seq_time += 10e-9 + 20e-9 + plateau # width + spacing + plateau

    
    def add_y_gate(self, angle):
        """
        Add an y gate with specific angle to the sequence
        """

        pulse_amp_name = "Control Pulse - Amplitude #" + str(self._next_pulse)
        pulse_plat_name = "Control Pulse - Plateau #" + str(self._next_pulse)
        pulse_phase_name = "Control Pulse - Phase #" + str(self._next_pulse)
        pulse_output_name = "Control Pulse - Output #" + str(self._next_pulse)

        plateau = self._get_time_for_angle(angle)

        self._meas_object.updateValue( pulse_amp_name, 1 )
        self._meas_object.updateValue( pulse_plat_name, plateau )
        self._meas_object.updateValue( pulse_phase_name, self._accumulated_z_phase + 90 )
        self._meas_object.updateValue( pulse_output_name, 0 )
    
        self._next_pulse += 1
        self._seq_time += 10e-9 + 20e-9 + plateau # width + spacing + plateau


    def finish_sequence(self):
        """
        Finish sequence so that it's ready for measurement
        """
        self._meas_object.updateValue( "Control Pulse - # of pulses", self._next_pulse )

        self._add_measurement_pulse()
        self._add_first_pulse_delay()


    def _add_first_pulse_delay(self):
        """
        Caculate and add the first pulse delay
        """
        first_pulse_delay = self._start_time - self._seq_time + 10e-9 # add one width
        self._meas_object.updateValue( "Control Pulse - First pulse delay", first_pulse_delay )

    def _add_measurement_pulse(self):    
        """
        Add measurement pulse after the qubit pulse sequence
        """
        pulse_amp_name = "Control Pulse - Amplitude #" + str(self._next_pulse)
        pulse_plat_name = "Control Pulse - Plateau #" + str(self._next_pulse)
        pulse_phase_name = "Control Pulse - Phase #" + str(self._next_pulse)
        pulse_output_name = "Control Pulse - Output #" + str(self._next_pulse)


        self._meas_object.updateValue( pulse_amp_name, .175 )
        self._meas_object.updateValue( pulse_plat_name, 2e-6 )
        self._meas_object.updateValue( pulse_phase_name, 0 )
        self._meas_object.updateValue( pulse_output_name, 1 )

    def reset(self):
        """
        Reset parameters when a new sequence is started
        """
        self._meas_object.updateValue( "Control Pulse - # of pulses", 0 )
        self._next_pulse = 1
        self._seq_time = 0

    
        
class SQCFactory():
    """
    Factory class that creates the single qubit pulse object depending on 
    the measurement and pulse type
    """

    @staticmethod
    def get_single_qubit_controller(meas_object, meas_type, pulse_type = "ARBITRARY"):
        """
        Create a single qubit pulse object depending on the measurement 
        and pulse type
        """
        
        if meas_type == MEAS_TYPE_SIMULATION:
            return SQCSim(meas_object, pulse_type)
        elif meas_type == MEAS_TYPE_EXPERIMENT:
            return SQCExp(meas_object, pulse_type)

        