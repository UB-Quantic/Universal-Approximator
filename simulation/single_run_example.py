from universal_approximator import UniversalApproximator
import matplotlib.pyplot as plt
import numpy as np

# Parameters of the gates to be computed
# 2 layers
# p = [[ 7.30775879e-01, -5.11802708e-02, -1.53394120e-06], \
#      [ 4.90298425e-01,  8.36578751e-01,  5.36512016e-01]]

# 3 layers
# p = [[-5.00333011e-05,  7.83699572e-01,  4.31687529e-01], \
#      [ 2.66722653e+00,  2.61685260e-03,  7.07511546e-01], \
#      [-1.33359853e+00,  6.07931285e-06,  3.57390521e-02]]

# 5 layers
p = [[ 5.70445799e+00,  1.57081148e+00,  1.22220816e+00], \
     [ 2.77603782e+00,  2.90030600e-06, -4.35489567e-01], \
     [ 2.14084701e+00,  7.98600609e-01, -8.56527131e-05], \
     [ 7.45353020e-01,  7.72153830e-01,  5.17595767e-01], \
     [ 1.45124672e+00,  7.85470816e-01,  6.55444925e-01]]

# Parameters for tanh

# 2 layers:
# p = [[ 2.82388717e-01, -7.92945098e-05,  2.84881410e+00], \
#      [ 5.24087654e-01,  1.57071607e+00,  8.97491417e-01]]

# 3 layers:
# p = [[ 2.59814459e-01, -8.40594852e-06,  1.77668041e+00], \
#      [ 5.19644894e-01,  6.98617036e-01, -1.01408773e+00], \
#      [-1.24415605e-05,  1.15313816e+00,  3.60325230e-01]]

# 4 layers:
# p = [[-2.59803483e-01, -1.68474316e-05,  1.36497562e+00], \
#      [ 5.19571864e-01,  3.26053638e-01,  6.17190448e-01], \
#      [ 2.75606996e-05,  9.90624509e-01,  6.94996320e-01], \
#      [-4.99341659e-06,  4.64907779e-01,  9.65668603e-01]]     


# 5 layers:
# p = [[ 0.5535071,  -0.096717,    2.50623578], \
#      [ 0.0159662,  -0.37351418,  1.43809871], \
#      [ 0.54754098,  0.91932381,  1.03097342], \
#      [ 0.53828845,  0.71573941, -2.11100829], \
#      [ 0.27341608, -0.55536742,  0.15635837]]

for i, p_line in enumerate(p): p[i] = [x*2 for x in p_line] # This is due to a convention in the algorithm
from scipy.optimize import minimize


if __name__ == "__main__":

     # x values to be evaluated     
     x = np.linspace( -np.pi, np.pi, num=21)

     features = {
          "n_averages_meas": 5e3,
          "n_averages_cal": 5e3,
          "spacing": 10e-9, # in terms of width
          "width": 5e-9, # s 
          "power_qubit_pulse": 15, #dBm
          # "truncation_range": 3, #n sigmas for gaussian trunc.range
          # "calib_point": 51,
          "reset": False,
          "reset_cav_amp": 157.75e-3,
          "reset_qub_amp": 4.5e-3,
          "drag_scaling": -523.19163e-12
     }

     # Create the UniversalApproximator object and update parameters
     univ_app = UniversalApproximator(n_layers=1, \
          measurement_type="EXPERIMENT", pulse_type="GAUSSIAN",
          features = features)

     univ_app.set_x_range(x)
     univ_app.define_function( lambda x: (1 + np.cos(x) ) / 2 )
     p = [.95, 0, 0]

# coseno de menos pi a pi, con una sola capa
     
     results = minimize(univ_app.chi_square, p, method='Nelder-Mead', options={"disp": True})

#      pass
     print(results)

     univ_app.update_param(p)

     print()
     P_1 = [] 
     theta = []
     
     # f= open("tanh_ok_9w.txt","w+")

     for x_i in x:
          univ_app.update_x(x_i)
          P_0 = univ_app.run()

          # Result is P0 and we want P1. We add it to the array.
          P_1.append( 1 - P_0 )
          theta.append(univ_app.theta)
          print("For x =", x_i, ", Pe is ",1 - P_0)
          # f.write(x_i, 1-P_0)
     print()

     # theta_desv, theta_max = [], []
     # for theta_list in theta:
     #      desv_sum = 0
     #      for theta_i in theta_list:
     #           theta_i_corrected = theta_i
     #           while theta_i_corrected < 0:
     #                theta_i_corrected += 2*np.pi
     #           while theta_i_corrected >= 2*np.pi:
     #                theta_i_corrected -= 2*np.pi

     #           desv_sum += abs(theta_i_corrected)
     #      theta_desv.append(desv_sum)
     #      theta_max.append( np.amax(theta_list) )


     # f.close()
     # Plot results
     plt.figure(1)
     plt.plot(x,P_1)
     # plt.figure(2)
     # plt.plot(x,theta_desv)
     # plt.figure(3)
     # plt.plot(x,theta_max)
     plt.show()


     np.savetxt("last_measure.txt", np.stack((x, P_1)))
     # plt.figure(2)
     # diff = []
     # for x_i, pi in zip(x,P_1):
     #      diff_i = (1 + np.tanh(x_i)) / 2 - pi
     #      diff.append(diff_i)
     # plt.plot(x,abs(diff))
     

