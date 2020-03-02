import numpy as np

def get_time_for_angle(angle):
    """
    Obtain the pulse time for a desired angle rotation
    """
    w = 130706947.3895477
    phi_0 = 1.2997382639681208
    
    while angle < phi_0:
        angle = angle + 2*np.pi
    pulse_time = (angle - phi_0) / w

    return pulse_time

p = [[ 0.5535071,  -0.096717,    2.50623578], \
     [ 0.0159662,  -0.37351418,  1.43809871], \
     [ 0.54754098,  0.91932381,  1.03097342], \
     [ 0.53828845,  0.71573941, -2.11100829], \
     [ 0.27341608, -0.55536742,  0.15635837]]

# Let's see for x = -3:
# 
# The sequence is Y(po * x + p1 ) -> Z(p2)


x=-3
theta_y, theta_z = [],[]
for p_i in p:
    theta_y_i = p_i[0] * x + p_i[1]
    theta_z_i = p_i[2]
    theta_y.append( theta_y_i )
    theta_z.append( theta_z_i )

print(theta_y)
print(theta_z)

plateaus = []
accumulated_phase = []

plateau_0 = get_time_for_angle(theta_y[0])
plateau_1 = get_time_for_angle(theta_y[1])
plateau_2 = get_time_for_angle(theta_y[2])
plateau_3 = get_time_for_angle(theta_y[3])
plateau_4 = get_time_for_angle(theta_y[4])
plateaus = [ plateau_0, plateau_1, plateau_2, plateau_3, plateau_4]

phase_0 = np.rad2deg(theta_z[0]) + 90
phase_1 = np.rad2deg(theta_z[1])
phase_2 = np.rad2deg(theta_z[2])
phase_3 = np.rad2deg(theta_z[3])
phase_4 = np.rad2deg(theta_z[4])

accumulated_phase = [phase_0, \
                     phase_0 + phase_1, \
                     phase_0 + phase_1 + phase_2, \
                     phase_0 + phase_1 + phase_2 + phase_3, \
                     phase_0 + phase_1 + phase_2 + phase_3 + phase_4, \
                    ]

# print(plateaus)
# print(accumulated_phase)

A = 0.00010330938870216828
c = 0.00030081969676472867


seq_time = 10e-9 * 6 + 20e-9 * 5 + plateau_0 + plateau_1 + plateau_2 + plateau_3 + plateau_4
first_pulse_delay = 1e-6 - seq_time
# print(first_pulse_delay)

#result is 245uV
result = 380e-6
P0 = ( result - (c - A) ) / (2 * A)
# print(P0)
