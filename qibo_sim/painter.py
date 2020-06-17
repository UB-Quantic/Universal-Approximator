import os
import matplotlib.pyplot as plt
import numpy as np

files = os.listdir()
values = []
success = []
evals = []
#Scipy
for file in ['lbfgsb_%s.out'%l for l in range(1, 21)]:
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if 'fun' in line:
            line = line.split(':')
            values.append(np.float(line[1][:-1]))
        elif 'nfev' in line:
            line = line.split(':')
            evals.append(int(line[1][:-2]))
        elif 'success' in line:
            line = line.split(':')
            if 'True' in line[1][1:-1]:
                success.append(1)
            else:
                success.append(0)


fig, axs = plt.subplots(nrows=2)
axs[0].scatter(np.arange(1, 21), values, c=success)
axs[0].set(yscale='log', ylim = [0.000001,1])
axs[1].scatter(np.arange(1, 21), evals, c=success)
axs[1].set(yscale='log')
plt.show()
'''

#SGD
values = []
for file in ['sgd_1_%s.out'%l for l in range(1, 21)]:
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        print(line[0])
        if line[0] == '(':
            break
    line = line.split(',')[0][1:]
    values.append(np.float(line))

fig, axs = plt.subplots()
axs.scatter(np.arange(1, 21), values)
axs.set(yscale='log', ylim = [0.000001,1])

plt.show()'''