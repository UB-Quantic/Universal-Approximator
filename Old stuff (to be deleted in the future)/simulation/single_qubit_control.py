# Author: David López-Núñez <dln492@gmail.com>
import os
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from Labber import ScriptTools

MEAS_TYPE_SIMULATION = "SIMULATION"
MEAS_TYPE_EXPERIMENT = "EXPERIMENT"
PULSE_TYPE_GAUSS_PLAT = "GAUSSIAN_PLATEAU"
PULSE_TYPE_GAUSSIAN = "GAUSSIAN"


def general_cosinus(t, w, phi, A, c):
    return A * np.cos(w * t + phi) + c

class SingleQubitControl(ABC):
    """
    A class that controls the qubit
    """
    def __init__(self, meas_object, pulse_type, cal_file, features):

        self._meas_object = meas_object
        self._pulse_type = pulse_type

        self._calibration_file = cal_file
        self._calibration_file_result = "Cal_Res.hdf5"

        # Prepares calibration measurement to be done
        relPath = os.path.dirname(os.path.abspath(__file__))
        self._calibration_meas = ScriptTools.MeasurementObject(
                os.path.join( relPath, self._calibration_file),
                os.path.join( relPath, self._calibration_file_result ) )


        self.sq_exec = SQExFactory.get_single_qubit_controller(\
                    meas_object, pulse_type, features)

        self.sq_exec.apply_features(features, self._calibration_meas)

    def calibrate(self):
        """
        Calibrate both the qubit pulses and the qubit measurement
        """
        # calibration done
        # self._w = 130706947.3895477
        # self._phi_0 = 1.2997382639681208
        # self._A = 0.00010330938870216828
        # self._c = 0.00030081969676472867
        # return
        # Perform calibration measurement
        ( x_cal, y_cal ) = self._calibration_meas.performMeasurement()

        # Adjust oscillations to cosinus and retrieve values
        self._w, self._phi_0, self._A, self._c = self._fit_cosinus( x_cal, y_cal )
        self.sq_exec.introduce_calibration(self._w, self._phi_0, self._A, self._c)

    def process_result(self, result):
        """
        Process result and convert it into P_0
        """
        P0 = ( result - (self._c - self._A) ) / (2 * self._A)
        return P0

    def _fit_cosinus(self, x_cal, y_cal):
        """
        Fit the calibration result to a general cosinus
        """
        # First some trials are obtained so that the fit is succesful
        if isinstance(y_cal[0], complex):
            y_cal = np.absolute(y_cal)
        max_y = np.amax(y_cal)
        min_y = np.amin(y_cal)
        A_trial = abs(max_y - min_y) / 2
        c_trial = (max_y + min_y) / 2

        # w and phi_0 are kind of known in advance
        param_ini = [self.sq_exec.expected_w, self.sq_exec.expected_phi_0,\
                     A_trial, c_trial]
        parameters, par_errors = curve_fit( general_cosinus, x_cal, y_cal, param_ini)

        # mapping parameters
        w = parameters[0]
        phi = parameters[1]
        A = parameters[2]
        c = parameters[3]

        # Plot results in case you want to be sure of the calibration
        # plt.figure(1)
        # plt.plot(x_cal,y_cal)
        # y_sim = []
        # y_trial = []
        # for t in x_cal:
        #     y_sim.append(\
        #         general_cosinus(t, w, phi, A, c))
        #     # y_trial.append(\
        #     #     general_cosinus(t, self.sq_exec.expected_w, self.sq_exec.expected_phi_0, \
        #     #     A_trial, c_trial))

        # plt.plot(x_cal, y_sim)
        # # plt.plot(x_cal, y_trial)
        # plt.show()
        return w, phi, A, c

    def _get_time_for_angle(self, angle):
        """
        Obtain the pulse time for a desired angle rotation
        """

        while angle < self._phi_0:
            angle = angle + 2*np.pi
        pulse_time = (angle - self._phi_0) / self._w

        return pulse_time

    def add_x_gate(self, angle):
        """
        Add an x gate with specific angle to the sequence
        """
        self.sq_exec.add_x_gate(angle)

    def add_y_gate(self, angle):
        """
        Add an y gate with specific angle to the sequence
        """
        self.sq_exec.add_y_gate(angle)

    def add_z_gate(self, angle):
        """
        Add an z gate with specific angle to the sequence
        """
        self.sq_exec.add_z_gate(angle)

    def finish_sequence(self):
        """
        Finish sequence so that it's ready for measurement
        """
        self.sq_exec.prepare_measurement()

    def reset(self):
        """
        Reset parameters when a new sequence is started
        """
        self.sq_exec.reset()

    def add_pre_reset(self):
        """
        Add pre-reset protocol por cooling the qubit
        """
        self.sq_exec.add_pre_reset()



class SQExFactory():
    """
    Factory class that creates the single qubit pulse executer object depending
    on the pulse type
    """

    @staticmethod
    def get_single_qubit_controller(meas_object, pulse_type, features):
        """
        Create a single qubit pulse executer object depending on the pulse type
        """

        if pulse_type == PULSE_TYPE_GAUSS_PLAT:
            return SQExGaussPlat(meas_object, features)
        elif pulse_type == PULSE_TYPE_GAUSSIAN:
            return SQExGaussian(meas_object, features)
            # return SQCExp(meas_object, pulse_type)


class SQExecuter(ABC):
    """
    Single Qubit Executer for creating pulse sequences depending on the pulse
    type
    """

    def __init__(self, meas_object, features):
        self._meas_object = meas_object
        self._w, self._phi_0, self._A, self._c = 0,0,0,0
        self._accum_z_phase = 0
        self._next_pulse = 1
        self._seq_time = 0
        self._features = features


    def apply_features(self, features, cal_object):
        """
        Apply desired features
        """
        n_averages_meas_name = "Digitizer - Number of averages"
        if "n_averages_meas" in features:
            self._meas_object.updateValue( n_averages_meas_name, features["n_averages_meas"] )

        n_averages_cal_name = "Digitizer - Number of averages"
        if "n_averages_cal" in features:
            cal_object.updateValue( n_averages_cal_name, features["n_averages_cal"] )

        if "spacing" in features:
            self._spacing = features["spacing"]

        width_cal_name = "Control Pulse - Width #1"
        if "width" in features:
            self._width = features["width"]
            cal_object.updateValue( width_cal_name, features["width"])

        power_qubit_pulse_name = "Qubit Pulse Power"
        if "power_qubit_pulse" in features:
            self._meas_object.updateValue( power_qubit_pulse_name, features["power_qubit_pulse"] )
            cal_object.updateValue( power_qubit_pulse_name, features["power_qubit_pulse"] )

        n_averages_cal_name = "Digitizer - Number of averages"
        if "n_averages_cal" in features:
            cal_object.updateValue( n_averages_cal_name, features["n_averages_cal"] )

        truncation_range_name = "Control Pulse - Truncation range"
        if "truncation_range" in features:
            self._meas_object.updateValue( truncation_range_name, features["truncation_range"] )
            cal_object.updateValue( truncation_range_name, features["truncation_range"] )

        calib_point_name = "Control Pulse - Amplitude # 1 - # of points"
        if "calib_point" in features:
            cal_object.updateValue( calib_point_name, features["calib_point"] )
        
        drag_scaling_name = "Control Pulse - DRAG scaling"
        if "drag_scaling" in features:
            self._meas_object.updateValue( drag_scaling_name, features["drag_scaling"] )
            cal_object.updateValue( drag_scaling_name, features["drag_scaling"] )


        reset_pulse_qub_amp_name = "Control Pulse - Amplitude #1" 
        reset_pulse_cav_amp_name = "Control Pulse - Amplitude #2"
        if features["reset"] == True:
            cal_object.updateValue( reset_pulse_qub_amp_name, features["reset_qub_amp"] )
            cal_object.updateValue( reset_pulse_cav_amp_name, features["reset_cav_amp"] )



    def introduce_calibration(self, w, phi_0, A, c):
        """
        Introduce calibration values
        """
        self._w = w
        self._phi_0 = phi_0
        self._A = A
        self._c = c

    @abstractmethod
    def add_x_gate(self, angle):
        """
        Add x gate of phase "phase"
        """
        pass

    @abstractmethod
    def add_y_gate(self, angle):
        """
        Add y gate of phase "phase"
        """
        pass

    def add_z_gate(self, angle):
        """
        Add z gate of phase "phase"
        """
        # Z gates are actually virtual Z gates, so they are only accumulated phase
        # in next x-y pulses
        self._accum_z_phase += np.rad2deg( angle )

        while self._accum_z_phase < 0:
            self._accum_z_phase += 360
        while self._accum_z_phase >= 360:
            self._accum_z_phase -= 360

    def prepare_measurement(self):
        """
        Prepare for measurement
        """
        self._add_measurement_pulse()
        # self._add_first_pulse_delay()

    @abstractmethod
    def _add_measurement_pulse(self):
        """
        Add measurement pulse after the qubit pulse sequence
        """
        pass

    @abstractmethod
    def _add_first_pulse_delay(self):
        """
        Caculate and add the first pulse delay
        """
        pass

    @abstractmethod
    def add_pre_reset(self):
        """
        Add pre-reset protocol por cooling the qubit
        """
        pass

    def reset(self):
        """
        Reset for another measurement
        """
        self._meas_object.updateValue( "Control Pulse - # of pulses", 0 )
        self._next_pulse = 1
        self._seq_time = 0
        self._accum_z_phase = 0


class SQExGaussPlat(SQExecuter):
    """
    Single Qubit pulse executer class with Gaussian pulses with plateau
    """

    def __init__(self, meas_object, features):
        self.expected_w = 25e6 * 2*np.pi
        self.expected_phi_0 = 0.1
        self._meas_time = 1e-6
        SQExecuter.__init__(self, meas_object, features)

    def _get_plateau_for_angle(self, angle):
        """
        Obtain the pulse time for a desired angle rotation
        """

        while angle < self._phi_0:
            angle = angle + 2*np.pi
        plateau_time = (angle - self._phi_0) / self._w

        return plateau_time

    def add_x_gate(self, angle):
        """
        Add an x gate with specific angle to the sequence
        """

        pulse_amp_name = "Control Pulse - Amplitude #" + str(self._next_pulse)
        pulse_plat_name = "Control Pulse - Plateau #" + str(self._next_pulse)
        pulse_phase_name = "Control Pulse - Phase #" + str(self._next_pulse)
        pulse_output_name = "Control Pulse - Output #" + str(self._next_pulse)

        plateau = self._get_plateau_for_angle(angle)

        self._meas_object.updateValue( pulse_amp_name, 1 )
        self._meas_object.updateValue( pulse_plat_name, plateau )
        self._meas_object.updateValue( pulse_phase_name, self._accum_z_phase )
        self._meas_object.updateValue( pulse_output_name, 0 )

        self._meas_object.updateValue( "Control Pulse - # of pulses", self._next_pulse )
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

        plateau = self._get_plateau_for_angle(angle)

        self._meas_object.updateValue( pulse_amp_name, 1 )
        self._meas_object.updateValue( pulse_plat_name, plateau )
        if self._accum_z_phase + 90 >= 360:
            phase = self._accum_z_phase + 90 - 360
        else:
            phase = self._accum_z_phase + 90
        self._meas_object.updateValue( pulse_phase_name, phase )
        self._meas_object.updateValue( pulse_output_name, 0 )

        self._meas_object.updateValue( "Control Pulse - # of pulses", self._next_pulse )
        self._next_pulse += 1
        self._seq_time += 10e-9 + 20e-9 + plateau # width + spacing + plateau

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

        self._meas_object.updateValue( "Control Pulse - # of pulses", self._next_pulse )

    def _add_first_pulse_delay(self):
        """
        Caculate and add the first pulse delay
        """
        first_pulse_delay = self._meas_time - self._seq_time + 10e-9 # add one width
        self._meas_object.updateValue( "Control Pulse - First pulse delay", first_pulse_delay )

    def add_pre_reset(self):
        """
        Add pre-reset protocol por cooling the qubit
        """
        pass

class SQExGaussian(SQExecuter):
    """
    Single Qubit pulse executer class with Gaussian pulses modulated by amplitude
    """

    def __init__(self, meas_object, features):
        self.expected_w = 2*np.pi
        self.expected_phi_0 = 0
        self._meas_time = 5e-6
        self._width = 15e-9
        self._spacing = 2 * self._width
        self._mod_freq = 200e6
        SQExecuter.__init__(self, meas_object, features)

    def _get_amplitude_for_angle(self, angle):
        """
        Get the amplitude necessary for the desired rotation
        """
        while angle < self._phi_0 - np.pi:
            angle += 2*np.pi
        while angle >= self._phi_0 + np.pi:
            angle -= 2*np.pi
        amplitude = (angle - self._phi_0) / self._w
        return amplitude

    def add_x_gate(self, angle):
        """
        Add x gate of phase "phase"
        """
        pulse_amp_name = "Control Pulse - Amplitude #" + str(self._next_pulse)
        pulse_width_name = "Control Pulse - Width #" + str(self._next_pulse)
        pulse_plat_name = "Control Pulse - Plateau #" + str(self._next_pulse)
        pulse_spac_name = "Control Pulse - Spacing #" + str(self._next_pulse)
        pulse_phase_name = "Control Pulse - Phase #" + str(self._next_pulse)
        pulse_mod_freq_name = "Control Pulse - Mod. frequency #" + str(self._next_pulse)
        pulse_output_name = "Control Pulse - Output #" + str(self._next_pulse)

        amplitude = self._get_amplitude_for_angle(angle)

        self._meas_object.updateValue( pulse_amp_name, amplitude )
        self._meas_object.updateValue( pulse_width_name, self._width )
        self._meas_object.updateValue( pulse_plat_name, 0 )
        self._meas_object.updateValue( pulse_spac_name, self._spacing )
        self._meas_object.updateValue( pulse_phase_name, self._accum_z_phase )
        self._meas_object.updateValue( pulse_mod_freq_name, self._mod_freq )
        self._meas_object.updateValue( pulse_output_name, 0 )

        self._meas_object.updateValue( "Control Pulse - # of pulses", self._next_pulse )
        self._next_pulse += 1
        self._seq_time += self._width + self._spacing # width + spacing + plateau

    def add_y_gate(self, angle):
        """
        Add y gate of phase "phase"
        """
        pulse_amp_name = "Control Pulse - Amplitude #" + str(self._next_pulse)
        pulse_width_name = "Control Pulse - Width #" + str(self._next_pulse)
        pulse_plat_name = "Control Pulse - Plateau #" + str(self._next_pulse)
        pulse_spac_name = "Control Pulse - Spacing #" + str(self._next_pulse)
        pulse_phase_name = "Control Pulse - Phase #" + str(self._next_pulse)
        pulse_mod_freq_name = "Control Pulse - Mod. frequency #" + str(self._next_pulse)
        pulse_output_name = "Control Pulse - Output #" + str(self._next_pulse)

        amplitude = self._get_amplitude_for_angle(angle)

        self._meas_object.updateValue( pulse_amp_name, amplitude )
        self._meas_object.updateValue( pulse_width_name, self._width )
        self._meas_object.updateValue( pulse_plat_name, 0 )
        self._meas_object.updateValue( pulse_spac_name, self._spacing )
        if self._accum_z_phase + 90 >= 360:
            phase = self._accum_z_phase + 90 - 360
        else:
            phase = self._accum_z_phase + 90
        self._meas_object.updateValue( pulse_phase_name, phase )
        self._meas_object.updateValue( pulse_mod_freq_name, self._mod_freq )
        self._meas_object.updateValue( pulse_output_name, 0 )

        self._meas_object.updateValue( "Control Pulse - # of pulses", self._next_pulse )
        self._next_pulse += 1
        self._seq_time += self._width + self._spacing # width + spacing + plateau

    def _add_measurement_pulse(self):
        """
        Add measurement pulse after the qubit pulse sequence
        """
        pulse_amp_name = "Control Pulse - Amplitude #" + str(self._next_pulse)
        pulse_width_name = "Control Pulse - Width #" + str(self._next_pulse)
        pulse_plat_name = "Control Pulse - Plateau #" + str(self._next_pulse)
        pulse_spac_name = "Control Pulse - Spacing #" + str(self._next_pulse - 1) # changed the previous spacing!
        pulse_phase_name = "Control Pulse - Phase #" + str(self._next_pulse)
        pulse_mod_freq_name = "Control Pulse - Mod. frequency #" + str(self._next_pulse)
        pulse_output_name = "Control Pulse - Output #" + str(self._next_pulse)


        self._meas_object.updateValue( pulse_amp_name, .175 )
        self._meas_object.updateValue( pulse_width_name, 10e-9 )
        self._meas_object.updateValue( pulse_plat_name, 2e-6 )
        self._meas_object.updateValue( pulse_spac_name, 40e-9 )
        self._meas_object.updateValue( pulse_phase_name, 0 )
        self._meas_object.updateValue( pulse_mod_freq_name, 70e6 )
        self._meas_object.updateValue( pulse_output_name, 1 )

        self._meas_object.updateValue( "Control Pulse - # of pulses", self._next_pulse )

    def _add_first_pulse_delay(self):
        """
        Caculate and add the first pulse delay
        """
        first_pulse_delay = self._meas_time - self._seq_time + self._width # add two width
        self._meas_object.updateValue( "Control Pulse - First pulse delay", first_pulse_delay )

    def add_pre_reset(self):
        """
        Add pre-reset protocol por cooling the qubit
        """
        # Add cavity pulse
        pulse_amp_name = "Control Pulse - Amplitude #" + str(self._next_pulse)
        pulse_width_name = "Control Pulse - Width #" + str(self._next_pulse)
        pulse_plat_name = "Control Pulse - Plateau #" + str(self._next_pulse)
        pulse_spac_name = "Control Pulse - Spacing #" + str(self._next_pulse)
        pulse_phase_name = "Control Pulse - Phase #" + str(self._next_pulse)
        pulse_mod_freq_name = "Control Pulse - Mod. frequency #" + str(self._next_pulse)
        pulse_output_name = "Control Pulse - Output #" + str(self._next_pulse)

        self._meas_object.updateValue( pulse_amp_name, self._features["reset_qub_amp"] )
        self._meas_object.updateValue( pulse_width_name, self._width )
        self._meas_object.updateValue( pulse_plat_name, 2e-6 )
        self._meas_object.updateValue( pulse_spac_name, -2e-6 )
        self._meas_object.updateValue( pulse_phase_name, 0 )
        self._meas_object.updateValue( pulse_mod_freq_name, self._mod_freq )
        self._meas_object.updateValue( pulse_output_name, 0 )

        self._meas_object.updateValue( "Control Pulse - # of pulses", self._next_pulse )
        self._next_pulse += 1
        self._seq_time += self._width # width + spacing + plateau
        
        # Add qubit pulse
        pulse_amp_name = "Control Pulse - Amplitude #" + str(self._next_pulse)
        pulse_width_name = "Control Pulse - Width #" + str(self._next_pulse)
        pulse_plat_name = "Control Pulse - Plateau #" + str(self._next_pulse)
        pulse_spac_name = "Control Pulse - Spacing #" + str(self._next_pulse)
        pulse_phase_name = "Control Pulse - Phase #" + str(self._next_pulse)
        pulse_mod_freq_name = "Control Pulse - Mod. frequency #" + str(self._next_pulse)
        pulse_output_name = "Control Pulse - Output #" + str(self._next_pulse)

        self._meas_object.updateValue( pulse_amp_name, self._features["reset_cav_amp"] )
        self._meas_object.updateValue( pulse_width_name, self._width )
        self._meas_object.updateValue( pulse_plat_name, 2e-6 )
        self._meas_object.updateValue( pulse_spac_name, 2e-6 )
        self._meas_object.updateValue( pulse_phase_name, 0 )
        self._meas_object.updateValue( pulse_mod_freq_name, 70e6 )
        self._meas_object.updateValue( pulse_output_name, 1 )

        self._meas_object.updateValue( "Control Pulse - # of pulses", self._next_pulse )
        self._next_pulse += 1
        self._seq_time += self._width + 4e-6 # width + spacing + plateau