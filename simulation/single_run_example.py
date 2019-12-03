from universal_approximator import UniversalApproximator

x = -1
p = [[ 7.30775879e-01 -5.11802708e-02 -1.53394120e-06] \
     [ 4.90298425e-01  8.36578751e-01  5.36512016e-01]]

ua = UniversalApproximator()
ua.update_param(p)
ua.update_x(x)
pol = ua.run()