# Basic imports
import numpy as np
import matplotlib.pyplot as plt

# Own code imports
import universal_approximant as ua

# The program starts here
if __name__ == "__main__":

    # Define the x domain and the noise
    x = np.linspace( -10, 10, num=101 )
    features = { "sim_noise" : 0 } 

    # Create the Universal Approximant class
    univ_app = ua.UniversalApproximant( ua.SIMULATION, 5, features )
    
    # Introduce parameters
    p = [ 0.5535071,  -0.096717,    2.50623578, \
         0.0159662,  -0.37351418,  1.43809871, \
         0.54754098,  0.91932381,  1.03097342, \
         0.53828845,  0.71573941, -2.11100829, \
         0.27341608, -0.55536742]

    univ_app.update_param(p)
    univ_app.define_range(x)

    res = univ_app.run() # returns P0
    pe = [1-x for x in res]

    plt.plot(x, pe)
    plt.show()
