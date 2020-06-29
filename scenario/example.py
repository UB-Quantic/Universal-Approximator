# Basic imports
import numpy as np
import matplotlib.pyplot as plt
import time as t
import os
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

    name_file = _RELPATH_PARENT+"\\New Results\\experimental_results.json"
    with open(name_file, 'r') as f:
        param_dict = json.load(f)
    # p = param_dict["tanh"]["2L"]["x"][:-1]
    p = param_dict["tanh"]["10k 7n 15p chi Powell"]["2L"]["x"]


    # Create the Universal Approximant class
    univ_app = ua.UniversalApproximant( ua.EXPERIMENT, 2, exp_features )    
    x = np.linspace( -10, 10, num=31 )
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
    # plt.plot(x, pe_the, label="The." )
    plt.plot(x, tanh(x), label="tanh")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Pe")
    plt.show()

    log_name = "Exp_Tanh_2L 10k 7n 15p chi Powell.txt"
    np.savetxt(log_name, np.stack((x, pe)) )
    # np.savetxt(log_name, np.stack((x, pe_exp, pe_the)) )
    # np.savetxt(log_name, np.stack((x, pe, simple_sinusodial_function(x))) )

