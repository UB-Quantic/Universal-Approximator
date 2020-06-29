# Basic imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time as t

# Own code imports
import universal_approximant as ua

# Here you can define all your functions that you want to obtain
def simple_sinusodial_function(x):
    return ( (1 + np.cos(x + np.pi) ) / 2 )

def tanh(x):
    return ( (1 + np.tanh(x) ) / 2 )

def relu(x):
    return np.clip(x, 0, np.max(x))


# The program starts here
if __name__ == "__main__":

    # Define the x domain and the noise
    # x = np.array( [-10,-8,-6,-4,-2,-1,-0.5,0,0.5,1,2,4,6,8,10] )
    x = np.linspace( -1,1,num=41 )
    cal_rate = 0
    sim_features = { "sim_noise" : 0 } 
    exp_features = {
        "reset": {
            "isEnabled" : True,
            "qub_ampl": 5e-3,
            "cav_ampl": .200
        },
        "drag": {
            "isEnabled" : True,
            "drag_scaling": -523.19163e-12
        }
    }
    # Create the Universal Approximant class
    univ_app = ua.UniversalApproximant( ua.EXPERIMENT, 4, exp_features )
    
    # Introduce initial parameters if wanted
    p = [0, 0, 0, 0, 0,0,0,0,0,0,0]

    # Prepare Universal Approximant class
    univ_app.update_param(p)
    univ_app.define_range(x)
    univ_app.define_function(relu)
    univ_app.define_cal_rate(cal_rate)

    time_start = t.perf_counter()

    # Mimimize!
    results = minimize(univ_app.chi_square, p, method='Powell', options={"disp": True})

    time_elapsed = (t.perf_counter() - time_start)
    print("\"time\": ", time_elapsed)
    print("\"fun\": ", results.fun)
    print("\"message\": ", results.message)
    print("\"nfev\": ", results.nfev)
    print("\"nit\":", results.nit)
    print("\"success\":", results.success)
    print("\"x\":", results.x)


    univ_app.update_param(results.x)
    data = univ_app.run()
    pe = [1-x for x in data]

    plt.plot( x, pe)
    plt.plot( x, tanh(x))

    plt.show()
    log_name = "Optimization.txt"
    np.savetxt(log_name, np.stack((x, data)) )

    # Delete result files
    for filename in glob.glob("./res*.hdf5"):
        os.remove(filename) 
