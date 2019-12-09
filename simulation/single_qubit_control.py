# Author: David López-Núñez <dln492@gmail.com> 
import os
import numpy as np
from scipy.optimize import curve_fit

from Labber import ScriptTools


def polarization_cosinus(t, w, phi):
    return np.cos(w * t + phi)


class SingleQubitControl():
    """
    A class that controls the qubit
    """
    def __init__(self, meas_object, pulse_type = "ARBITRARY", calibration_file = "Simulation_Calibration.hdf5"):
        self._meas_object = meas_object
        self._pulse_type = pulse_type
        
        relPath = os.path.dirname(os.path.abspath(__file__))
        self._calibration_meas = ScriptTools.MeasurementObject(
                os.path.join( relPath, calibration_file),
                os.path.join( relPath, "Calibration_Result.hdf5" ) )

        self._w, self._phi_0 = 0,0
        self._accumulated_z_phase, self._next_pulse = 0,1 # initialize variables

    
    def calibrate(self):

        ( t_cal, pol_cal ) = self._calibration_meas.performMeasurement()
        self._w, self._phi_0 = self._fit_sinus( t_cal, pol_cal )

    
    def _fit_sinus(self, t_cal, pol_cal):
        param_ini = [25e6 * 2*np.pi, 0.1]
        parameters, par_errors = curve_fit( polarization_cosinus, t_cal, pol_cal, param_ini)

        w = parameters[0]
        phi = parameters[1]

        return w, phi


    def _get_time_for_angle(self, angle):
        
        if angle < self._phi_0:
            angle = angle + 2*np.pi
        pulse_time = (angle - self._phi_0) / self._w

        return pulse_time


    def add_x_gate(self, angle):
        pulse_plat_name = "Control Pulse - Plateau #" + str(self._next_pulse)
        phase_plat_name = "Control Pulse - Phase #" + str(self._next_pulse)

        self._meas_object.updateValue( pulse_plat_name, self._get_time_for_angle(angle) )
        self._meas_object.updateValue( phase_plat_name, self._accumulated_z_phase )
    
        self._next_pulse += 1

    
    def add_y_gate(self, angle):
        pulse_plat_name = "Control Pulse - Plateau #" + str(self._next_pulse)
        phase_plat_name = "Control Pulse - Phase #" + str(self._next_pulse)

        self._meas_object.updateValue( pulse_plat_name, self._get_time_for_angle(angle) )
        self._meas_object.updateValue( phase_plat_name, self._accumulated_z_phase + 90 )
    
        self._next_pulse += 1


    def add_z_gate(self, angle):
        self._accumulated_z_phase += np.rad2deg( angle )


    def finish_sequence(self):
        self._meas_object.updateValue( "Control Pulse - # of pulses", self._next_pulse - 1 )


    def reset(self):
        self._meas_object.updateValue( "Control Pulse - # of pulses", 0 )
        self._next_pulse = 1
        


        
