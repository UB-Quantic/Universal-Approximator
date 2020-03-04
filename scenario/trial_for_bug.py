# Basic imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Own code imports
import universal_approximant as ua

# Here you can define all your functions that you want to obtain
def simple_sinusodial_function(x):
    return ( (1 + np.cos(x + np.pi) ) / 2 )
def relu(x):
    return x * (x > 0)

# The program starts here
if __name__ == "__main__":

    # Define the x domain and the noise
    x = np.linspace( -1, 1, num=101 )
    features = { "sim_noise" : 0 } 

    # Create the Universal Approximant class
    univ_app = ua.UniversalApproximant( ua.SIMULATION, 2, features )
    
    # Introduce initial parameters if wanted
    p = [0.56367295, 0.70850466, 0.60167699, 0.58892208, 0.39282808]

    # Prepare Universal Approximant class
    univ_app.update_param(p)
    univ_app.define_range(x)
    univ_app.define_function(relu)

    # Mimimize!
    for i in range(10):
        result = univ_app.chi_square(p)
        print(result)
    # results = minimize(univ_app.chi_square, p, method='Powell', options={"disp": True})

    # print(results)