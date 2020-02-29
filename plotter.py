import numpy as np
import matplotlib.pyplot as plt

step = np.loadtxt("step_DRAG_m1_5w_50av_reset.txt")
tanh = np.loadtxt("tanh_DRAG_m1_5w.txt")
tanh_sim = np.loadtxt("tanh_sim_5l.txt")

plt.plot(tanh_sim[0], tanh_sim[1])
# plt.plot(tanh[0], tanh[1])
# plt.plot(step[0], step[1])
plt.show()