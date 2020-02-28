from Labber import Scenario
import numpy as np


class ScenarioCreator():
    """
    Class that creates the scenario for Labber
    """
    def __init__(self):

        self._scn = Scenario()
        self._instruments = {}

    def add_instrument(self, instr_info):
        instr = self._scn.add_instrument( \
            instr_info["instrument"], name = instr_info["name"] )
        self._instruments[instr_info["name"]] = instr
        
        self.update_instrument(instr_info["name"], instr_info["features"])
        
    def update_instrument(self, instr_name, features):

        instr = self._instruments[instr_name]        
        for feature in features:
            instr.values[feature["name"]] = feature["value"]


    def add_signal_connection(self, signal_r, signal_d):
        signal_receiver = signal_r["instr_name"] + " - " + signal_r["signal"]
        signal_doner = signal_d["instr_name"] + " - " + signal_d["signal"]
        self._scn.add_connection( signal_doner, signal_receiver)

    def add_step(self, instr_ch, step_range):
        channel = instr_ch["name"] + " - " + instr_ch["channel"]
        self._scn.add_step( channel, step_range )

    def add_log(self, instr_ch):
        channel = instr_ch["name"] + " - " + instr_ch["channel"]
        self._scn.add_log(channel)

    def set_log_name(self, log_name):
        self._scn.log_name = log_name

    def save_log(self, save_name):
        self._scn.save(save_name)



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

def instr_ch(instrument_name, channel):
    intr_ch = {
        "name": instrument_name,
        "channel": channel
    }
    return instr_ch

def signal(instrument_name, signal):
    signal = {
        "instr_name": instrument_name,
        "signal": signal
    }
    return signal


if __name__ == "__main__":
    scn = ScenarioCreator()

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
    scn.add_instrument( ssg_info )

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
    scn.add_instrument( sqpg_info )

    # Add and configure Single-Qubit Pulse Generator
    sqs_features = [
        # Qubit
        feat("Delta", 4.825e9),
        feat("Epsilon", 0),
        feat("Drive frequency", 5.025e9),
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
        feat("Noise, Delta 1", 100e6),
        feat("Noise, Epsilon 1", 0),
        feat("Noise, Drive 1", 0),
        feat("High frequency cut-off 1", 20e9)
    ]
    sqs_info = instr_info( "Single-Qubit Simulator", "Qubit Simulator", sqs_features )
    scn.add_instrument( sqs_info )


    # add signal connections between qubit simulator and other instruments
    scn.add_signal_connection(
        signal("Qubit Simulator", "Trace - I"), signal("Control Pulse", "Trace - I")  )
    scn.add_signal_connection(
        signal("Qubit Simulator", "Trace - Q"), signal("Control Pulse", "Trace - Q")  )
    scn.add_signal_connection(
        signal("Qubit Simulator", "Noise trace - Epsilon"), signal("Signal", "Signal")  )
    scn.add_signal_connection(
        signal("Qubit Simulator", "Noise trace - Delta"), signal("Signal", "Signal")  )

    # add the first pulse sequence
    sqpg_pulse_feat = [
        feat("Width #1", 5e-9),
        feat("Plateau #1", 0),
        feat("Spacing #1", 0),
        feat("Phase #1", 0),
        feat("Mod. frequency #1", 200e6),
        feat("Ratio I/Q #1", 1),
        feat("Phase diff. #1", 0)
    ]
    scn.update_instrument( sqpg_info["name"], sqpg_pulse_feat)

    # now add step
    scn.add_step( instr_ch("Control Pulse", "Amplitude #1") , np.linspace(-.5, .5, 501) )

    # add log channels
    scn._scn.add_log("Qubit Simulator - Polarization - Z")

    scn.set_log_name('Test Rabi')
    scn.save_log("test_rabi")

