# This module will handle the dirty work of working with Labber Scenario class

# Basic imports
import numpy as np
# import os

# Labber imports
from Labber import Scenario
from Labber import config
# from Labber import ScriptTools

# own code imports
# here we basically import the constants
SIMULATION = "SIMULATION"
EXPERIMENT = "EXPERIMENT"

# Define some nice constants
CAL = "calibration"
ALG = "algorithm"

# Define useful functions
def feat(name, value):
    feature = { 
        "name": name,
        "value": value
         }
    return feature

def instr_info(instrument, name, features):
    instr_info = {
        "instrument": instrument,
        "name": name,
        "features": features
    }  
    return instr_info




class ScenarioManager():
    """
    This is the hard-working class that creates the measurement contexts so that
    you can do your stuff
    """

    def __init__(self, meas_type, n_layers, features):
        """
        It expects the following parameters:
            - meas_type: either "SIMULATION" or "EXPERIMENT"
            - n_layers: number of pulse layers (up to 5)
            - features: dict of features to be changed from default (read docs)
        """
        self._meas_type = meas_type
        self._n_layers = n_layers
        self._features = features

        self.cal_scn = Scenario()
        self.alg_scn = Scenario()
        self._scn_dict = {
            CAL: self.cal_scn,
            ALG: self.alg_scn
        }

        self._instruments = {}
        
        self._prepare_sim_calibration()
        self._prepare_sim_algorithm()

        self.add_features(features)
        self._finish_sim_calibration()

        self._accum_z_phase = 0

########################################################;
#				---EXTERNAL---;
########################################################;

    def set_x_range(self, x):
        self.x = x
        self.alg_scn.add_step( "Manual - Value 1", x )

    def generate_default_pulses(self, n_pulses):
        for i in range(n_pulses):
            sqpg_pulse_feat = [
                feat("Width #" + str(i+1), 5e-9),
                feat("Plateau #" + str(i+1), 0),
                feat("Spacing #" + str(i+1), 15e-9),
                feat("Phase #" + str(i+1), 90),
                feat("Mod. frequency #" + str(i+1), 70e6),
                feat("Ratio I/Q #" + str(i+1), 1),
                feat("Phase diff. #" + str(i+1), 0)
            ]
            self._update_instrument( ALG, "Control Pulse", sqpg_pulse_feat)
        number_of_pulses = [feat("# of pulses", n_pulses)]
        self._update_instrument(ALG, "Control Pulse", number_of_pulses)

    def add_lookup_tables(self, n_layers, A):
        for layer in range(n_layers):
            self._add_lookup_table(layer, [ x[layer] for x in A] )

    def add_virtual_zs(self, n_layers, theta):
        for layer in range(n_layers - 1):
            self._add_virtual_z(layer, theta[layer])    

    def save_algorithm(self):
        self._set_log_name(ALG, "Algorithm")
        self._save_log(ALG, "Algorithm")

    def add_features(self, features):
        if "sim_noise" in features:
            for scn in self._scn_dict:
                qubit_sim = self._instruments[ (scn, "Qubit Simulator") ]        
                signal = self._instruments[ (scn, "Signal") ]        
                qubit_sim.values["Noise, Delta 1"] = features["sim_noise"] * 20e6
                signal.values["Noise Ampltiude"] = features["sim_noise"] * 20e6

    def remove_steps(self):
        scn = self._scn_dict[ALG]
        for step_name in scn.step_names():
            if step_name.startswith("Control Pulse - Amplitude #"):
                scn.remove_step(step_name)

    def reset_phase(self):
        self._accum_z_phase = 0 

########################################################;
#				---INSTRUMENTS---;
########################################################;

    def _add_single_qubit_pulse_generator(self, scn):
        # Add and configure Single-Qubit Pulse Generator
        sqpg_features = [
            # Waveform features
            feat("Sequence",3),
            feat("Sample rate",1e9),
            feat("Number of points",1e3),
            feat("First Pulse Delay",30e-9),
            feat("Trim waveform to sequence",True),
            feat("Buffer start to restore size",False),
            feat("Number of outputs",0),
            feat("# of pulses",1),
            # Pulses setting -no individual pulses
            feat("Pulse type",0),
            feat("Truncation range",3),
            feat("Start at zero",False),
            feat("Edge-to-edge pulses",True),
            feat("Edge position",1),
            feat("Use SSB mixing",True),
            feat("Use DRAG",False),
            # Readout
            feat("Generate tomography pulse",False),
            feat("Generate readout",False),
            feat("Sample-and-hold readout",False),
            # Output
            feat("Swap IQ", False),
            feat("Add pre-pulses", False),
            feat("Generate gate", False)
        ]
        sqpg_info = instr_info( "Single-Qubit Pulse Generator", "Control Pulse", sqpg_features )
        self._add_instrument( scn, sqpg_info )

    def _add_single_qubit_simulator(self, scn):
        # Add and configure Single-Qubit Pulse Generator
        sqs_features = [
            # Qubit
            feat("Delta", 4.825e9),
            feat("Epsilon", 0),
            feat("Drive frequency", 4.895e9),
            feat("Drive relative to qubit frequency", False),
            feat("Drive amplitude", 250e6),
            feat("Drive type", 1),
            feat("Time step, simulation", 20e-12),
            feat("Use rotating frame", True),
            feat("Use rotating-wave approximation", True),
            feat("Time step, output traces", 1e-9),
            feat("Number of randomizations", 2),
            # Noise
            feat("Noise sources", 1),
            feat("Disable noise during pulses", False),
            feat("Noise type 1", 1),
            feat("Noise, Delta 1", 0),
            feat("Noise, Epsilon 1", 0),
            feat("Noise, Drive 1", 0),
            feat("High frequency cut-off 1", 20e9)
        ]
        sqs_info = instr_info( "Single-Qubit Simulator", "Qubit Simulator", sqs_features )
        self._add_instrument( scn, sqs_info )

    def _add_simple_signal_generator(self, scn):
        # Add and configure Simple Signal Generator:
        ssg_features = [
            feat("Frequency", 0),
            feat("Amplitude", 0),
            feat("Phase", 0),
            feat("Duration", 1),
            feat("Number of points", 1e3),
            feat("Add noise", True),
            feat("Noise amplitude", .1)
        ]
        ssg_info = instr_info( "Simple Signal Generator", "Signal", ssg_features )
        self._add_instrument( scn, ssg_info )

    def _add_manual_instrument(self, scn):
        # Add manual instrument
        manual_info = instr_info( "Manual", "Manual", {})
        self._add_instrument( scn, manual_info)

    def _add_awg(self, scn):
        # Add and configure Arbitrary Waveform Generator
        awg_features = [
            # Channels
            feat("Ch1 - Enabled",True),
            feat("Ch1 - Function",4),
            feat("Ch1 - Amplitude",1),
            feat("Ch1 - Offset",0),
            feat("Ch1 - Sweep rate",0),
            feat("Ch1 - Trig mode",1),
            feat("Ch1 - Cycles",1),
            feat("Ch2 - Enabled",True),
            feat("Ch2 - Function",4),
            feat("Ch2 - Amplitude",1),
            feat("Ch2 - Offset",0),
            feat("Ch2 - Sweep rate",0),
            feat("Ch2 - Trig mode",1),
            feat("Ch2 - Cycles",1),
            feat("Ch3 - Enabled",True),
            feat("Ch3 - Function",4),
            feat("Ch3 - Amplitude",1),
            feat("Ch3 - Offset",0),
            feat("Ch3 - Sweep rate",0),
            feat("Ch3 - Trig mode",1),
            feat("Ch3 - Cycles",1),
            feat("Ch4 - Enabled",True),
            feat("Ch4 - Function",4),
            feat("Ch4 - Amplitude",1),
            feat("Ch4 - Offset",0),
            feat("Ch4 - Sweep rate",0),
            feat("Ch4 - Trig mode",1),
            feat("Ch4 - Cycles",1),
            # Trigger
            feat("Trig I/O",0),
            feat("Trig Sync Mode",0),
            feat("Trig delay",0),
            feat("Waveform alignment",0),
            # Marker
            feat("Ch1 - Marker Mode",0),
            feat("Ch2 - Marker Mode",0),
            feat("Ch3 - Marker Mode",0),
            feat("Ch4 - Marker Mode",0)
        ]
        awg_info = instr_info( "Keysight PXI AWG", "Keysight PXI AWG", awg_features )
        self._add_instrument( scn, awg_info )

    def _add_hvi_trigger(self, scn):
        # Add and configure HVI trigger:
        hvi_features = [
            feat("Output",True),
            feat("Trig period",200e-6),
            feat("Digitizer delay",0),
            feat("Auto-detect",True),
        ]
        hvi_info = instr_info( "Keysight PXI HVI Trigger", "HVI Trigger", hvi_features )
        self._add_instrument( scn, hvi_info )

    def _add_digitizer(self, scn):
        # Add and configure Keysight PXI Digitizer:
        dig_features = [
            # Acquisition
            feat("FPGA output configuration",1),
            feat("Number of samples",4e3),
            feat("Number of records",1),
            feat("Number of averages",5e3),
            # Trigger
            feat("Trig I/O",0),
            feat("Trig Sync Mode",0),
            feat("Trig Mode",1),
            # Channels
            feat("Ch1 - Enabled",True),
            feat("Ch1 - Coupling",0),
            feat("Ch1 - Impedance",0),
            feat("Ch1 - Range",1),
            feat("Ch2 - Enabled",True),
            feat("Ch2 - Coupling",0),
            feat("Ch2 - Impedance",0),
            feat("Ch2 - Range",1),
            feat("Ch3 - Enabled",True),
            feat("Ch3 - Coupling",0),
            feat("Ch3 - Impedance",0),
            feat("Ch3 - Range",1),
            feat("Ch4 - Enabled",True),
            feat("Ch4 - Coupling",0),
            feat("Ch4 - Impedance",0),
            feat("Ch4 - Range",1),
            # Demod control
            feat("Skip Time",1e-6),
            feat("Integration time",2e-6),
            feat("Use phase reference signal",False),
            feat("Hide I/Q/REF values",False),
            feat("LO freq 1",70e-6)
        ]
        dig_info = instr_info( "Keysight PXI Digitizer FPGA Demod", "Digitizer", dig_features )
        self._add_instrument( scn, dig_info )


    def _add_quicksyn_rf_source(self, scn):
        # Add and configure QuickSyn rf source:
        qs_rf_features = [
            # Settings
            feat("Frequency", 7.8246e12),
            feat("Power",13),
            feat("Output",0),
            feat("Pulse modulation",1),
            # Reference Oscillator
            feat("Ref Source",1),
            feat("Output Reference Source",0)##
        ]
        qs_rf_info = instr_info( "QuickSyn Signal Generator", "QS RF Soruce", qs_rf_features )
        self._add_instrument( scn, qs_rf_info )

    def _add_rs_iq_source(self, scn):
        # Add and configure Rohde&Schwarz RF Source:
        rs_rf_features = [
            # Settings
            feat("Frequency", 4.8948e9),
            feat("Power",15),
            feat("Phase", 0),
            feat("Mode",0),
            feat("Automatic levelling control (ALC)",1),
            feat("Output",True),
            feat("Pulse modulation",0),
            feat("I/Q modulation",True),
            feat("I/Q wideband",True),
            feat("I/Q Adjustments",False),
            # Reference Oscillator
            feat("LO Source",1),
            feat("LO Frequency",0),
            feat("LO Output Source",2),
            feat("LO Output Frequency",0)
        ]
        rs_rf_info = instr_info( "Rohde&Schwarz RF Source", "RS IQ Source", rs_rf_features )
        self._add_instrument( scn, rs_rf_info )

    def _add_instrument(self, scn, instr_info):
        """
        Add instrument in an easier way
        """
        instr = self._scn_dict[scn].add_instrument( \
            instr_info["instrument"], name = instr_info["name"] )
        self._instruments[ (scn, instr_info["name"]) ] = instr
        
        self._update_instrument(scn, instr_info["name"], instr_info["features"])
        
    def _update_instrument(self, scn, instr_name, features):

        instr = self._instruments[ (scn, instr_name) ]        
        for feature in features:
            instr.values[feature["name"]] = feature["value"]

########################################################;
#				---SIMULATION---;
########################################################;

    def _prepare_sim_calibration(self):
        scn = CAL
        self._add_single_qubit_pulse_generator(scn)
        self._add_single_qubit_simulator(scn)
        self._add_simple_signal_generator(scn)

        self._create_sim_connections(scn)
        self._add_sim_cal_pulses(scn)
        
        self._add_sim_cal_step(scn)
        self._add_sim_log_channel(scn)

    def _finish_sim_calibration(self):
        scn = CAL
        self._set_log_name(scn, "Calibration")
        self._save_log(scn, "Calibration")

    def _prepare_sim_algorithm(self):
        scn = ALG
        self._add_single_qubit_pulse_generator(scn)
        self._add_single_qubit_simulator(scn)
        self._add_simple_signal_generator(scn)
        self._add_manual_instrument(scn)

        self._create_sim_connections(scn)
        self._add_sim_log_channel(scn)

    def _create_sim_connections(self, scn):
        # add signal connections between qubit simulator and other instruments
        current_scn = self._scn_dict[scn]
        current_scn.add_connection( \
            "Control Pulse - Trace - I", "Qubit Simulator - Trace - I"  )
        current_scn.add_connection( \
            "Control Pulse - Trace - Q", "Qubit Simulator - Trace - Q"  )
        current_scn.add_connection( \
            "Signal - Signal", "Qubit Simulator - Noise trace - Epsilon"  )
        current_scn.add_connection( \
            "Signal - Signal", "Qubit Simulator - Noise trace - Delta"  )
        
    def _add_sim_cal_pulses(self, scn):
        # add the first pulse sequence
        sqpg_pulse_feat = [
            feat("Width #1", 5e-9),
            feat("Plateau #1", 0),
            feat("Spacing #1", 0),
            feat("Phase #1", 0),
            feat("Mod. frequency #1", 70e6),
            feat("Ratio I/Q #1", 1),
            feat("Phase diff. #1", 0)
        ]
        self._update_instrument( scn, "Control Pulse", sqpg_pulse_feat)        

    def _add_sim_cal_step(self, scn):
        # now add step
        step_range = np.linspace(-.5, .5, 501)
        self._scn_dict[scn].add_step( "Control Pulse - Amplitude #1", step_range )

    def _add_sim_log_channel(self, scn):
        self._scn_dict[scn].add_log("Qubit Simulator - Polarization - Z")
        self._scn_dict[scn].add_log("Qubit Simulator - Trace - Pz")

########################################################;
#				---EXPERIMENT---;
########################################################;


########################################################;
#				---COMMON---;
########################################################;

    def _set_log_name(self, scn, name):
        self._scn_dict[scn].log_name = name
    
    def _save_log(self, scn, name):
        self._scn_dict[scn].save(name)

    def _add_lookup_table(self, layer, A):
        lookup_step = self.alg_scn.add_step(
            channel = "Control Pulse - Amplitude #" + str(layer + 1),
            single = 0.0 )

        lookup = config.lookup.LookUpTable(self.x, A)

        param = config.step.RelationParameter()
        param.variable = 'p1'
        param.channel_name = 'Manual - Value 1'
        param.use_lookup = True
        param.lookup = lookup 

        lookup_step.use_relations = True
        lookup_step.relation_parameters = [param]
        lookup_step.equation = 'p1'

    def _add_virtual_z(self, layer, theta):
        self._calc_accumulated_phase(theta)
        acc_phase = [ feat("Phase #" + str(layer+2), 90 + self._accum_z_phase) ]
        self._update_instrument( ALG, "Control Pulse", acc_phase)

    def _calc_accumulated_phase(self, theta):
        self._accum_z_phase += np.rad2deg( theta )

        while self._accum_z_phase < 0:
            self._accum_z_phase += 360
        while self._accum_z_phase >= 360:
            self._accum_z_phase -= 360
