from universal_approximator import UniversalApproximator
import matplotlib.pyplot as plt

# x values to be evaluated     
x = [-1.,-0.95918367,-0.91836735,-0.87755102,-0.83673469,-0.79591837,-0.75510204,-0.71428571,\
     -0.67346939,-0.63265306,-0.59183673,-0.55102041,-0.51020408,-0.46938776,-0.42857143,\
     -0.3877551 ,-0.34693878,-0.30612245,-0.26530612,-0.2244898 ,-0.18367347,-0.14285714,\
     -0.10204082,-0.06122449,-0.02040816, 0.02040816, 0.06122449, 0.10204082, 0.14285714,\
     0.18367347, 0.2244898 , 0.26530612, 0.30612245, 0.34693878, 0.3877551 , 0.42857143, \
     0.46938776, 0.51020408, 0.55102041, 0.59183673, 0.63265306, 0.67346939, 0.71428571, \
     0.75510204, 0.79591837, 0.83673469, 0.87755102, 0.91836735, 0.95918367, 1.]

# reduced version of x vaues
# x = [-1.,-0.75510204, -0.46938776, -0.26530612, 0.02040816,  0.26530612, 0.46938776, 0.75510204, 1.]


# Parameters of the gates to be computed
# 2 layers
p = [[ 7.30775879e-01, -5.11802708e-02, -1.53394120e-06], \
     [ 4.90298425e-01,  8.36578751e-01,  5.36512016e-01]]

# 3 layers
p = [[-5.00333011e-05,  7.83699572e-01,  4.31687529e-01], \
     [ 2.66722653e+00,  2.61685260e-03,  7.07511546e-01], \
     [-1.33359853e+00,  6.07931285e-06,  3.57390521e-02]]

# 5 layers
# p = [[ 5.70445799e+00,  1.57081148e+00,  1.22220816e+00], \
#      [ 2.77603782e+00,  2.90030600e-06, -4.35489567e-01], \
#      [ 2.14084701e+00,  7.98600609e-01, -8.56527131e-05], \
#      [ 7.45353020e-01,  7.72153830e-01,  5.17595767e-01], \
#      [ 1.45124672e+00,  7.85470816e-01,  6.55444925e-01]]


if __name__ == "__main__":

     for i, p_line in enumerate(p): p[i] = [x*2 for x in p_line] # This is due to a convention in the algorithm

     # Create the UniversalApproximator object and update parameters
     univ_app = UniversalApproximator(n_layers=3, measurement_type="EXPERIMENT")

     univ_app.update_param(p)

     print()
     P_1 = [] 
     
     for x_i in x:
          univ_app.update_x(x_i)
          P_0 = univ_app.run()

          # Result is P0 and we want P1. We add it to the array.
          P_1.append( 1 - P_0 )
          print("For x =", x_i, ", Pe is ",1 - P_0)
     print()

     # Plot results
     plt.figure(1)
     plt.plot(x,P_1)
     plt.show()
