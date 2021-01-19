# Basic imports
import numpy as np
import matplotlib.pyplot as plt
import time as t
import os,glob
import json
# Own code imports
import universal_approximant_Jan as ua
from get_optimal_params import get_optimal_params

_RELPATH_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def simple_sinusodial_function(x):
    return ( (1 + np.cos(x + np.pi) ) / 2 )

def tanh(x):
    return ( (1 + np.tanh(5*x) ) / 2 )

def relu(x):
    return np.clip(x, 0, np.max(x))

def step(x):
    return 1 * (x > 0)

# The program starts here
if __name__ == "__main__":
    # Define the features
    exp_features = {
        "reset": {
            "isEnabled" : False
        },
        "drag": {
            "isEnabled" : False
        }
    }

    fun_names = ["relu", "tanh", "poly", "step"]
    n_layers_list = [1,2,3,4,5,6]

    for fun_name in fun_names:
        for n_layers in n_layers_list:
            # Introduce parameters    
            p = get_optimal_params('Weighted',fun_name,n_layers)[0]
            x = np.linspace( -1, 1, num=31 )          

            # Create the Universal Approximant class
            univ_app = ua.UniversalApproximant( ua.EXPERIMENT, n_layers, exp_features )
            univ_app.define_range(x)
            univ_app.update_param(p)

            res = univ_app.run() # returns P0
            pg = [x for x in res]

            # plt.plot(x, pg, label="Exp." )
            # plt.plot(x, tanh(x), label="relu")
            # plt.legend()
            # plt.xlabel("x")
            # plt.ylabel("Pe")
            # plt.show()

            log_name = "The_" + fun_name + "_" + str(n_layers) + "L .txt"
            np.savetxt(log_name, np.stack((x, pg)) )
        
            # Delete result files
            for filename in glob.glob("./res*.hdf5"):
                os.remove(filename) 

