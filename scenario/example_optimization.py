# Basic imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, basinhopping

# Own code imports
import universal_approximant as ua

# Here you can define all your functions that you want to obtain
def simple_sinusodial_function(x):
    return ( (1 + np.cos(x + np.pi) ) / 2 )

def tanh(x):
    return ( (1 + np.tanh(x) ) / 2 )

def relu(x):
    return np.clip(x, 0, np.max(x))

layers=1
# The program starts here
if __name__ == "__main__":

    # Define the x domain and the noise
    x = np.linspace( -1, 1, num=21 )
    features = { "sim_noise" : 1 }

    # Create the Universal Approximant class
    univ_app = ua.UniversalApproximant( ua.SIMULATION, layers, features )
    
    # Introduce initial parameters if wanted
    p = list(np.random.rand(3 * layers - 1))

    # Prepare Universal Approximant class
    univ_app.update_param(p)
    univ_app.define_range(x)
    univ_app.define_function(tanh)

    # Mimimize!
    #results = minimize(univ_app.chi_square, p, method='l-bfgs-b', options={"disp": True})
    results = basinhopping(univ_app.chi_square, p, niter=2, disp=True)
    print(results)

    univ_app.update_param(results['x'])


    res = univ_app.run()  # returns P0
    pe = [1 - x for x in res]

    plt.plot(x, pe)
    plt.plot(x, tanh(x))
    plt.show()