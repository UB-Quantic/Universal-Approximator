import numpy as np
from circuits import *

def est_grad_1(theta, measurements, h, target_distribution, function='square'):
    lay, qubits, n = theta.shape
    layers = (lay - 1) // 2
    gradient = np.empty_like(theta)
    theta_plus = theta.copy()
    theta_minus = theta.copy()
    c = cost(theta, measurements, target_distribution, function=function)
    for l in range(lay):
        for q in range(qubits):
            for i in range(n):
                theta_plus[l, q, i] += h
                theta_minus[l, q, i] -= h
                gradient[l, q, i] = 1 / 2 / h *(cost(theta_plus, measurements, target_distribution, function=function) -\
                                            cost(theta_minus, measurements, target_distribution, function=function))
                theta_plus[l, q, i] -= h
                theta_minus[l, q, i] += h

    return gradient, c

def adam_step(theta, measurements, h, target_distribution, a, b1, b2, m, v, t, epsi=1e-6):
    g, c = est_grad_1(theta, measurements, h, target_distribution, function='square')
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g**2

    m_ = m / (1 - b1**t)
    v_ = v / (1 - b2**t)

    theta_new = theta - a * m_ / (np.sqrt(v_) + epsi)

    return theta_new, m, v, c

def adam_optimizer(theta, measurements, h, target_distribution, a, b1, b2):
    t = 1
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    best_theta = theta.copy()
    obj = 1 / np.sqrt(measurements) / 2**theta.shape[1]
    best_c = 1
    while t < 1000:
        theta, m, v, c = adam_step(theta, measurements, h, target_distribution, a, b1, b2, m, v, t)
        t+= 1
        if t%20 == 0: print(t, c)
        if c < best_c:
            best_theta = theta
            best_c = c
        if best_c < obj:
            best_theta = theta
            break

    print(best_c, best_theta)
    return best_theta
