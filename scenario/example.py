# Basic imports
import numpy as np
import matplotlib.pyplot as plt
import time as t
import os,glob
import json
# Own code imports
import universal_approximant as ua

_RELPATH_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def simple_sinusodial_function(x):
    return ( (1 + np.cos(x + np.pi) ) / 2 )

def tanh(x):
    return ( (1 + np.tanh(x) ) / 2 )

def relu(x):
    return np.clip(x, 0, np.max(x))

# The program starts here
if __name__ == "__main__":
    # Define the features
    # sim_features = { "sim_noise" : 0 } 
    exp_features = {
        "reset": {
            "isEnabled" : True,
            "qub_ampl": 5e-3,
            "cav_ampl": .2
        },
        "drag": {
            "isEnabled" : True,
            "drag_scaling": -523.19163e-12
        }
    }


    # Introduce parameters

    # name_file = _RELPATH_PARENT+"\\New Results\\experimental_results.json"
    name_file = _RELPATH_PARENT+"\\New Results\\theoretical_results.json"
    with open(name_file, 'r') as f:
        param_dict = json.load(f)
    # p = param_dict["tanh"]["2L"]["x"][:-1]

    n_layers_range = [5]
    functions_range = [relu]

    for function in functions_range:
        if function == relu:
            f_name = "relu"
            x = np.linspace( -1, 1, num=41 )
        else:
            f_name = "tanh"
            x = np.linspace( -10, 10, num=41 )

        for n_layers in n_layers_range:
            NL = str(n_layers) + "L" 
            # p = param_dict[f_name]["10k 7n 41p log Powell"][NL]["x"]
            # p = param_dict[f_name][NL]["x"][:-1]
            # p = [2.09231546,   0.77120458, -15.40389215,   0.34309537,   0.06610264,
            #     -0.23606795,   0.0220145,   -0.07638291,  -1.32583386,   0.2280512,
            #     0.05306048]
            p = [1.57080042e+00, -7.69543678e+00,  8.73013121e-02, -4.50284027e-06,
                 5.61631234e+00,  8.53702178e-01, -3.14159346e+00,  9.95691393e+00,
                -4.15056814e-01,  1.09326620e-06, -5.71742121e+00,  9.43407150e-01,
                -3.21123805e-07,  5.23764231e+00]
            # p = [x/2 for x in p]

            # Create the Universal Approximant class
            univ_app = ua.UniversalApproximant( ua.EXPERIMENT, n_layers, exp_features )
            univ_app.define_range(x)
            univ_app.update_param(p)
            # t0 = t.perf_counter()
            res = univ_app.run() # returns P0
            # t1 = t.perf_counter()
            pe = [1-x for x in res]

            # time = str(t1-t0)
            # print("Time used has been ", time[:5], "s")
            # univ_app.update_param(p_the)
            # res = univ_app.run() # returns P0
            # pe_the = [1-x for x in res]

            plt.plot(x, pe, label="Exp." )
            # # plt.plot(x, pe_the, label="The." )
            plt.plot(x, relu(x), label="relu")
            # plt.legend()
            # plt.xlabel("x")
            # plt.ylabel("Pe")
            plt.show()

            # log_name = "Exp_" + f_name + "_" + str(n_layers) + "L 10k 7n 41p chi Powell p0prev.txt"
            # log_name = "The_" + f_name + "_" + str(n_layers) + "L.txt"
            # log_name = "Exp_Tanh_4L 10k 7n 31p chi cal25 Powell.txt"
            # np.savetxt(log_name, np.stack((x, pe)) )
            # np.savetxt(log_name, np.stack((x, pe_exp, pe_the)) )
            # np.savetxt(log_name, np.stack((x, pe, simple_sinusodial_function(x))) )

    # Delete result files
    for filename in glob.glob("./res*.hdf5"):
        os.remove(filename) 

