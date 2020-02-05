import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def cosine(A_pulse, w, A, c ):
    return A * np.cos(w * A_pulse) + c


data = np.loadtxt('simulation/Analysis/Rabi oscillations NO DRAG.txt', skiprows=3)
# data = np.loadtxt('simulation/Analysis/Rabi w Amplitude for Phase Calibration Reset.txt', skiprows=3)
AData = data[0,:]
yData = data[1,:] 

vMax = np.amax(yData)
vMin = np.amin(yData)

ampl_trial = abs(vMax - vMin) / 2
noise_trial = vMin + ampl_trial
w_trial = 2 * np.pi

param_init = [w_trial , ampl_trial, noise_trial]
param, fidelity = curve_fit(cosine, AData , yData, param_init )

w = param[0]
print("Ramsey freq is ", str(w/(2*np.pi)))

# w * A_pi = pi => A_pi = pi / w
pi_pulse_A = np.pi / w
pi_2_pulse_A = np.pi / (2 * w)

print("Pi pulse amplitude is ", pi_pulse_A)
print("Pi/2 pulse amplitude is ", pi_2_pulse_A)


ySim = cosine(AData, *param)

plt.figure(1)
plt.plot( AData, yData )
plt.plot( AData, ySim )
plt.show()