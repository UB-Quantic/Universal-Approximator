# This is an example on how everyhting works.

# Basic imports
import numpy as np
import matplotlib.pyplot as plt

# Own code imports
import universal_approximant as ua

# Here you can define all your functions that you want to obtain
def simple_sinusodial_function(x):
    return ( (1 + np.cos(x) ) / 2 )

# The program starts here
if __name__ == "__main__":

    # Define the domain
    x = np.linspace( -10, 10, num=201 )

    features = { "sim_noise" : 0 } # Noise from 0 to as much as you want


    # Create the Universal Approximant class
    univ_app = ua.UniversalApproximant( ua.SIMULATION, 5, features )
    
    # Introduce parameters if wanted
    p = [ 0.5535071,  -0.096717,    2.50623578, \
         0.0159662,  -0.37351418,  1.43809871, \
         0.54754098,  0.91932381,  1.03097342, \
         0.53828845,  0.71573941, -2.11100829, \
         0.27341608, -0.55536742]

    univ_app.update_param(p)
    univ_app.define_range(x)

    # res = univ_app.run()
    data = univ_app.run()
    pe = [(1-x)/2 for x in data[1]]

    plt.plot(data[0], pe)
    plt.show()