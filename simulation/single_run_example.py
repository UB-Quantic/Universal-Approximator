from universal_approximator import UniversalApproximator
import matplotlib.pyplot as plt



if __name__ == "__main__":

     # x values to be evalutaed     
     x = [-1.,-0.95918367,-0.91836735,-0.87755102,-0.83673469,-0.79591837,-0.75510204,-0.71428571,\
          -0.67346939,-0.63265306,-0.59183673,-0.55102041,-0.51020408,-0.46938776,-0.42857143,\
          -0.3877551 ,-0.34693878,-0.30612245,-0.26530612,-0.2244898 ,-0.18367347,-0.14285714,\
          -0.10204082,-0.06122449,-0.02040816, 0.02040816, 0.06122449, 0.10204082, 0.14285714,\
          0.18367347, 0.2244898 , 0.26530612, 0.30612245, 0.34693878, 0.3877551 , 0.42857143, \
          0.46938776, 0.51020408, 0.55102041, 0.59183673, 0.63265306, 0.67346939, 0.71428571, \
          0.75510204, 0.79591837, 0.83673469, 0.87755102, 0.91836735, 0.95918367, 1.]

     # Parameters of the gates to be computed
     p = [[ 7.30775879e-01, -5.11802708e-02, -1.53394120e-06], \
          [ 4.90298425e-01,  8.36578751e-01,  5.36512016e-01]]
     p = p*2 # This is due to a convention in the algorithm

     # Create the UniversalApproximator object. We want for now only two layers
     univ_app = UniversalApproximator(n_layers=2)

     # We add the parameters of the algorithm
     univ_app.update_param(p)

     print()
     P_1 = [] # initialize the vector with the excited state (|1>) probabilities of the measurements

     # Let's start a loop for all the x's in the experiment
     for x_i in x:
          # update the new x to be evaluate and run the algorithm
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
