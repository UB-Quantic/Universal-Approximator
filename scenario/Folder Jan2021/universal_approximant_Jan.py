# The Universal Approximant class it's in this module (What a surprise)

# Basic imports
import numpy as np
import scipy as sc
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Labber imports
from Labber import ScriptTools

# Own code imports
import scenario_Jan as scn

# Definition of basic constants
SIMULATION = "SIMULATION"
EXPERIMENT = "EXPERIMENT"

_RELPATH = os.path.dirname(os.path.abspath(__file__))

# Calibration functions
def calib_cosinus(t, w, A, c):
    return A * np.cos(w * t)**2 + c

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
        # self._check_n_layers(n_layers)
        self._check_features(features)

        self._meas_type = meas_type
        self._n_layers = n_layers
        self._features = features

        self.p = []
        self.historic_p = []
        self.historic_res = []
        self.f = None
        self.x = None

        self.runs_wout_cal = 0
        self.cal_rate = 0

        self.scn_mng =  scn.ScenarioManager( meas_type, n_layers, features)

        self.cal_meas = ScriptTools.MeasurementObject(
                os.path.join( _RELPATH, "calibration.labber"), # Here we should write the name parametrically
                os.path.join( _RELPATH, "res_cal.hdf5" ) )
        self.calibrate(plot=False)

    def _check_meas_type(self, meas_type):
        if meas_type not in [SIMULATION, EXPERIMENT]:
            print("Measurement type not supported")
            exit()
    
    def _check_n_layers(self, n_layers):
        if n_layers not in range(1,6):
            print("Number of layers not supported")
            exit()

    def _check_features(self, features):
        # TO BE DONE PROPERLY
        print("Features checking still no supported. You should be doing well")

    def calibrate(self, plot=False):
        
        calibration_data = self.cal_meas.performMeasurement()

        param = self._fit_calib_cosinus(calibration_data, plot)
        self._w = param[0]
        self._A = param[1]  
        self._c = param[2]

    def _fit_calib_cosinus(self, data, plot):
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

        if plot == True:
            np.savetxt("calib.txt", np.stack((x_cal, y_cal)))
            plt.plot(x_cal, y_cal)
            sim = calib_cosinus(x_cal, *param)
            plt.plot(x_cal, sim)
            plt.show()

        return param

    def update_param(self, param):
        if len(param) != self._n_layers * 3:
            print("Wrong number of parameters")
            exit()
        
        p = np.zeros((self._n_layers, 3))
        for lay in range(self._n_layers):
            p[lay][0] = param[lay * 3]
            p[lay][1] = param[lay * 3 + 1]
            p[lay][2] = param[lay * 3 + 2]
        
        self.p = p

    def define_range(self, x):
        self.x = x
        self.scn_mng.set_x_range(x)

        if self.f is not None:
            self._create_function_value()

    def define_cal_rate(self,cal_rate):
        self.cal_rate = cal_rate

    def run(self):

        self.runs_wout_cal += 1
        if self.cal_rate != 0:
            if self.runs_wout_cal % self.cal_rate == 0:
                self.calibrate()

        self._reset_scenario()
        self._calc_thetas()
        self._calc_amplitudes()

        self._add_hadamard_gate()
        self._add_lookup_tables_z()
        self._add_y_gates()

        
        self._add_measurement()
        self._save_algorithm()

        self.alg_meas = ScriptTools.MeasurementObject(
                os.path.join( _RELPATH, "algorithm.labber"), # Here we should write the name parametrically
                os.path.join( _RELPATH, "res.hdf5" ) )
        algorithm_raw = self.alg_meas.performMeasurement()
        result = self._convert_result(algorithm_raw[1])
        return result
   
    def _calc_thetas(self):

        theta_y = np.zeros( self._n_layers, dtype=np.longdouble)
        theta_z = np.zeros((len(self.x), self._n_layers), dtype=np.longdouble)

        for i, x_i in enumerate(self.x):
            for layer in range(self._n_layers):
                theta_z[i, layer] = self.p[layer][0] + self.p[layer][1] * x_i 

        
        for layer in range(self._n_layers):
            theta_y[layer] = self.p[layer][2]

        self.theta_y = theta_y 
        self.theta_z = theta_z 

    def _calc_amplitudes(self):
        ampl = np.zeros( self._n_layers, dtype=np.longdouble)
        for i in range(self._n_layers):
            ampl[i] = self._get_amplitude_for_angle(\
                self.theta_y[i])
        self.ampl = ampl

    def _get_amplitude_for_angle(self, angle):
        """
        Get the amplitude necessary for the desired rotation
        """
        while angle < - np.pi:
            angle += 2.0*np.pi
        while angle >=  np.pi:
            angle -= 2.0*np.pi

        # while angle < 0:
        #     angle += 2*np.pi
        # while angle >=  2*np.pi:
        #     angle -= 2*np.pi

        amplitude = angle / (self._w * 2)
        return amplitude

    def _add_lookup_tables_z(self):
        self.scn_mng.add_lookup_tables(self._n_layers, self.theta_z)

    def _add_hadamard_gate(self):
        self.scn_mng.add_hadamard_gate(self._get_amplitude_for_angle(np.pi/2))

    def _add_y_gates(self):
        self.scn_mng.add_y_gates(self._n_layers, self.ampl)

    def _save_algorithm(self):
        self.scn_mng.save_algorithm()

    def define_function(self,f):
        self.f = f
        if self.x is not None:
            self._create_function_value()

    def _create_function_value(self):
        self._fx = self.f(self.x)

    def chi_square(self, p):

        self.historic_p.append(p)
        self.update_param(p)
        P_1 = []
        P_0 = self.run()
        P_1 = [1 - x for x in P_0 ]
        result = ( 0.5 / len(self.x) ) * np.sum( ( P_1 - self._fx )**2 )
        self.historic_res.append(result)
        # plt.plot(self.x,P_1)
        # plt.plot(self.x,self._fx)
        # plt.show()
        return result

    def log_diff(self, p):
        self.update_param(p)
        P_1 = []
        P_0 = self.run()
        P_1 = [1 - x for x in P_0 ]
        result = ( 0.5 / len(self.x) ) * np.sum( ( P_1 - self._fx )**2 )
        log_result = np.log(result)
        # plt.plot(self.x,P_1)
        # plt.plot(self.x,self._fx)
        # plt.show()
        return log_result

    def _convert_result(self, raw_result):
        if isinstance(raw_result[0], complex):
            raw_result = np.absolute(raw_result)
        P0 = ( raw_result - (self._c ) ) / (self._A)
        return P0

    def _reset_scenario(self):
        self.scn_mng.remove_steps()
        self.scn_mng.reset_phase()

    def _add_measurement(self):
        self.scn_mng.add_measurement()


