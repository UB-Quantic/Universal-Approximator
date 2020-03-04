# Basic imports
import numpy as np
import matplotlib.pyplot as plt

# Own code imports
import universal_approximant as ua
from example_optimization import relu

# The program starts here
if __name__ == "__main__":

    # Define the x domain and the noise
    x = np.linspace( -1, 1, num=101 )
    features = { "sim_noise" : 0 } 

    # Create the Universal Approximant class
    univ_app = ua.UniversalApproximant( ua.SIMULATION, 2, features )
    
    # Introduce parameters
    '''p = [ 0.5535071,  -0.096717,    2.50623578, \
         0.0159662,  -0.37351418,  1.43809871, \
         0.54754098,  0.91932381,  1.03097342, \
         0.53828845,  0.71573941, -2.11100829, \
         0.27341608, -0.55536742]'''

    p = [0.56367295, 0.70850466, 0.60167699, 0.58892208, 0.39282808]
    univ_app.update_param(p)
    univ_app.define_range(x)

    for i in range(10):
        res = univ_app.run() # returns P0
        pe = [1-x for x in res]

        plt.plot(x, pe)
    plt.plot(x, relu(x), color='k')
    plt.show()
