from universal_approximator import UniversalApproximator



if __name__ == "__main__":

     x = 0.91836735
     p = [[ 7.30775879e-01, -5.11802708e-02, -1.53394120e-06], \
          [ 4.90298425e-01,  8.36578751e-01,  5.36512016e-01]]

     p_3 = [[-5.00333011e-05,  7.83699572e-01,  4.31687529e-01], \
            [ 2.66722653e+00,  2.61685260e-03,  7.07511546e-01], \
            [-1.33359853e+00,  6.07931285e-06,  3.57390521e-02]]

     ua = UniversalApproximator(n_layers=3)
     ua.update_param(p_3)
     ua.update_x(x)
     P_e = ua.run()
     print(P_e)