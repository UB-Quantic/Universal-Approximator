# Basic imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Own code imports
import universal_approximant as ua

# Here you can define all your functions that you want to obtain
def simple_sinusodial_function(x):
    return ( (1 + np.cos(x + np.pi) ) / 2 )

# The program starts here
if __name__ == "__main__":

    # Define the x domain and the noise
    x = np.linspace( -1, 1, num=101 )
    features = { "sim_noise" : 0 } 

    # Create the Universal Approximant class
    univ_app = ua.UniversalApproximant( ua.SIMULATION, 1, features )
    
    # Introduce initial parameters if wanted
    p = [0, 0]

    # Prepare Universal Approximant class
    univ_app.update_param(p)
    univ_app.define_range(x)
    univ_app.define_function(simple_sinusodial_function)

    # Mimimize!
    results = minimize(univ_app.chi_square, p, method='Powell', options={"disp": True})

    print(results)