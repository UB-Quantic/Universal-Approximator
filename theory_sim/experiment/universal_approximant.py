# The Universal Approximant class it's in this module (What a surprise)

# Basic imports
import numpy as np
import os
from scipy.optimize import curve_fit

# Labber imports
from Labber import ScriptTools

# Own code imports
import scenario as scn

# Definition of basic constants
SIMULATION = "SIMULATION"
EXPERIMENT = "EXPERIMENT"

_RELPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Calibration functions
def calib_cosinus(t, w, A, c):
    return A * np.cos(w * t) + c

def general_cosinus(t, w, phi, A, c):
    return A * np.cos(w * t + phi) + c




# THE CLASS
class UniversalApproximant():
    """
    Implements the universal approximant alogorithm for some given parameters
    """

    def __init__(self, meas_type, n_layers, features ):
        """
        It expects the following parameters:
            - meas_type: either "SIMULATION" or "EXPERIMENT"
            - n_layers: number of pulse layers (up to 5)
            - features: dict of features to be changed from default (read docs)
        """
        self._check_meas_type(meas_type)
        self._check_n_layers(n_layers)
        self._check_features(features)

        self._meas_type = meas_type
        self._n_layers = n_layers
        self._features = features

        self.p = []

        self.scn_mng =  scn.ScenarioManager( meas_type, n_layers, features)

        self.cal_meas = ScriptTools.MeasurementObject(
                os.path.join( _RELPATH, "Calibration.labber"), # Here we should write the name parametrically
                os.path.join( _RELPATH, "res.hdf5" ) )
        calibration_raw = self.cal_meas.performMeasurement()
        self.calibrate(calibration_raw)

    def _check_meas_type(self, meas_type):
        if meas_type not in [SIMULATION, EXPERIMENT]:
            print("Measurement type not supported")
            exit()
        
        # TO BE DELETED WHEN IMPLEMENTED
        if meas_type == EXPERIMENT:
            print("Experiment not yet implemented")
            exit()
    
    def _check_n_layers(self, n_layers):
        if n_layers not in range(1,6):
            print("Number of layers not supported")
            exit()

    def _check_features(self, features):
        # TO BE DONE PROPERLY
        print("Features checking still no supported. You should be doing well")

    def calibrate(self, calibration_data, plot=False):

        param = self._fit_calib_cosinus(calibration_data)
        self._w = param[0]
        self._A = param[1]
        self._c = param[2]

        if plot == True:
            print("Plotting option not yet done. Sorry :(")
            # TO BE DONE

    def _fit_calib_cosinus(self, data):
        x_cal = data[0]
        y_cal = data[1]
        # First some trials are obtained so that the fit is succesful
        if isinstance(y_cal[0], complex):
            y_cal = np.absolute(y_cal)
        max_y = np.amax(y_cal)
        min_y = np.amin(y_cal)
        A_trial = abs(max_y - min_y) / 2
        c_trial = (max_y + min_y) / 2

        w_trial = 2 * np.pi # TO BE DONE PROPERLY, NOT BY HAND
        param_ini = [w_trial, A_trial, c_trial]
        param, param_errors = curve_fit( calib_cosinus, x_cal, y_cal, param_ini)

        return param

    def update_param(self, param):
        if len(param) != self._n_layers * 3 - 1:
            print("Wrong number of parameters")
            exit()
        
        p = np.zeros((self._n_layers, 3))
        for lay in range(self._n_layers):
            p[lay][0] = param[lay * 3]
            p[lay][1] = param[lay * 3 + 1]
            if lay != self._n_layers - 1:
                p[lay][2] = param[lay * 3 + 2]
            else:
                p[lay][2] = 0
        
        self.p = p

    def define_range(self, x):
        self.x = x
        self.scn_mng.set_x_range(x)

    def run(self):
        self._calc_thetas()
        self._calc_amplitudes()

        self._generate_default_pulses()
        self._add_lookup_tables()
        self._add_virtual_z()

        self._save_algorithm()

        self.alg_meas = ScriptTools.MeasurementObject(
                os.path.join( _RELPATH, "Algorithm.labber"), # Here we should write the name parametrically
                os.path.join( _RELPATH, "res.hdf5" ) )
        algorithm_raw = self.alg_meas.performMeasurement()
        return algorithm_raw
   
    def _calc_thetas(self):
        theta_y = np.zeros((len(self.x), self._n_layers))
        theta_z = np.zeros(self._n_layers)

        for i, x_i in enumerate(self.x):
            for layer in range(self._n_layers):
                theta_y[i, layer] = self.p[layer][0] * x_i + self.p[layer][1]
        
        for layer in range(self._n_layers):
            theta_z[layer] = self.p[layer][2]

        self.theta_y = theta_y 
        self.theta_z = theta_z 

    def _calc_amplitudes(self):
        ampl = np.zeros((len(self.x), self._n_layers))
        for i in range(len(self.x)):
            for j in range(self._n_layers):
                ampl[i][j] = self._get_amplitude_for_angle(\
                    self.theta_y[i][j])
        self.ampl = ampl

    def _get_amplitude_for_angle(self, angle):
        """
        Get the amplitude necessary for the desired rotation
        """
        while angle < - np.pi:
            angle += 2*np.pi
        while angle >=  np.pi:
            angle -= 2*np.pi
        amplitude = angle / self._w
        return amplitude

    def _generate_default_pulses(self):
        self.scn_mng.generate_default_pulses(self._n_layers)

    def _add_lookup_tables(self):
        self.scn_mng.add_lookup_tables(self._n_layers, self.ampl)

    def _add_virtual_z(self):
        self.scn_mng.add_virtual_zs(self._n_layers, self.theta_z)

    def _save_algorithm(self):
        self.scn_mng.save_algorithm()









