# Author: David López-Núñez <dln492@gmail.com> 
import os
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


class SingleQubitControl():
    """
    A class that controls the qubit
    """
    def __init__(self, meas_object, pulse_type = "ARBITRARY", measurement_type = MEAS_TYPE_SIMULATION):
        self._meas_object = meas_object
        self._pulse_type = pulse_type

        if measurement_type == MEAS_TYPE_SIMULATION:
            calibration_file = SIM_CAL_TEMPLATE
            calibration_file_result = SIM_CAL_RESULT
        elif measurement_type == MEAS_TYPE_EXPERIMENT:
            calibration_file = EXP_CAL_TEMPLATE
            calibration_file_result = EXP_CAL_RESULT
        
        relPath = os.path.dirname(os.path.abspath(__file__))
        self._calibration_meas = ScriptTools.MeasurementObject(
                os.path.join( relPath, calibration_file),
                os.path.join( relPath, calibration_file_result ) )

        self._w, self._phi_0, self._A, self._c = 0,0,0,0
        self._accumulated_z_phase, self._next_pulse = 0,1 # initialize variables

    
    def calibrate(self):

        ( t_cal, y_cal ) = self._calibration_meas.performMeasurement()
        self._w, self._phi_0, self._A, self._c = self._fit_cosinus( t_cal, y_cal )
    
    def process_result(self, result):
        P0 = ( result - (c - A) ) / (2 * A) 

    
    def _fit_cosinus(self, t_cal, y_cal):
        max_y = np.amax(y_cal)
        min_y = np.amin(y_cal)
        A_trial = abs(max_y - min_y) / 2
        c_trial = (max_y + min_y) / 2

        param_ini = [25e6 * 2*np.pi, 0.1, 1, 0]
        parameters, par_errors = curve_fit( general_cosinus, t_cal, y_cal, param_ini)

        w = parameters[0]
        phi = parameters[1]
        A = parameters[2]
        c = parameters[3]

             # Plot results
        plt.figure(1)
        plt.plot(t_cal,y_cal)
        y_sim = []
        y_trial = []
        for t in t_cal:
            y_sim.append(\
                general_cosinus(t, w, phi, A, c))
            y_trial.append(\
                general_cosinus(t, 25e6 * 2*np.pi, 0.1, 1, 0))

        plt.plot(t_cal, y_sim)
        plt.plot(t_cal, y_trial)
        plt.show()


        return w, phi, A, c


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
        


        
class SingleQubitControlExp():
    """
    A class that controls the qubit
    """
    def __init__(self, meas_object, pulse_type = "ARBITRARY", measurement_type = MEAS_TYPE_SIMULATION):
        self._meas_object = meas_object
        self._pulse_type = pulse_type

        if measurement_type == MEAS_TYPE_SIMULATION:
            calibration_file = SIM_CAL_TEMPLATE
            calibration_file_result = SIM_CAL_RESULT
        elif measurement_type == MEAS_TYPE_EXPERIMENT:
            calibration_file = EXP_CAL_TEMPLATE
            calibration_file_result = EXP_CAL_RESULT
        
        relPath = os.path.dirname(os.path.abspath(__file__))
        self._calibration_meas = ScriptTools.MeasurementObject(
                os.path.join( relPath, calibration_file),
                os.path.join( relPath, calibration_file_result ) )

        self._w, self._phi_0, self._A, self._c = 0,0,0,0
        self._accumulated_z_phase, self._next_pulse = 0,1 # initialize variables

        self._seq_time = 0
        self._start_time = 1e-6

    
    def calibrate(self):

        #calibration done previously
        # self._w  = 123087093.12264751
        # self._phi_0  = 1.2341265752072634
        # self._A  = 0.00010792764573665441
        # self._c  = 0.0003037085096952924

        ( t_cal, y_cal ) = self._calibration_meas.performMeasurement()
        self._w, self._phi_0, self._A, self._c = self._fit_cosinus( t_cal, np.abs(y_cal) )
        # print(self._w, self._phi_0, self._A, self._c)
    
    def process_result(self, result):
        P0 = ( result - (self._c - self._A) ) / (2 * self._A)
        return P0

    
    def _fit_cosinus(self, t_cal, y_cal):
        max_y = np.amax(y_cal)
        min_y = np.amin(y_cal)
        A_trial = abs(max_y - min_y) / 2
        c_trial = (max_y + min_y) / 2

        param_ini = [20e6 * 2*np.pi, 0.1, A_trial, c_trial]
        parameters, par_errors = curve_fit( general_cosinus, t_cal, y_cal, param_ini)

        w = parameters[0]
        phi = parameters[1]
        A = parameters[2]
        c = parameters[3]

        # # Plot results
        # plt.figure(1)
        # plt.plot(t_cal,y_cal)
        # y_sim = []
        # # y_trial = []
        # for t in t_cal:
        #     y_sim.append(\
        #         general_cosinus(t, w, phi, A, c))
        #     # y_trial.append(\
        #     #     general_cosinus(t, 25e6 * 2*np.pi, 0.1, A_trial, c_trial))

        # plt.plot(t_cal, y_sim)
        # # plt.plot(t_cal, y_trial)
        # plt.show()

        return w, phi, A, c


    def _get_time_for_angle(self, angle):
        
        if angle < self._phi_0:
            angle = angle + 2*np.pi
        pulse_time = (angle - self._phi_0) / self._w

        return pulse_time


    def add_x_gate(self, angle):

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


    def add_z_gate(self, angle):
        self._accumulated_z_phase += np.rad2deg( angle )


    def finish_sequence(self):
        self._meas_object.updateValue( "Control Pulse - # of pulses", self._next_pulse )


        pulse_amp_name = "Control Pulse - Amplitude #" + str(self._next_pulse)
        pulse_plat_name = "Control Pulse - Plateau #" + str(self._next_pulse)
        pulse_phase_name = "Control Pulse - Phase #" + str(self._next_pulse)
        pulse_output_name = "Control Pulse - Output #" + str(self._next_pulse)



        self._meas_object.updateValue( pulse_amp_name, .175 )
        self._meas_object.updateValue( pulse_plat_name, 2e-6 )
        self._meas_object.updateValue( pulse_phase_name, 0 )
        self._meas_object.updateValue( pulse_output_name, 1 )

        first_pulse_delay = self._start_time - self._seq_time + 10e-9 # add one width
        self._meas_object.updateValue( "Control Pulse - First pulse delay", first_pulse_delay )

    
        # self._next_pulse += 1        


    def reset(self):
        self._meas_object.updateValue( "Control Pulse - # of pulses", 0 )
        self._next_pulse = 1
        self._seq_time = 0

        


        