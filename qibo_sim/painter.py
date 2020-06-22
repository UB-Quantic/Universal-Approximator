import os
import matplotlib.pyplot as plt
import numpy as np

files = os.listdir()
values = []
success = []
evals = []
name = 'cobyla'
for file in [name + '_%s.out'%l for l in range(1, 21)]:
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if 'fun' in line:
            line = line.split(':')
            print(line)
            try: values.append(np.float(line[1][:-1]))
            except:
                values.append(values[-1])
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
print(values)
axs[0].scatter(np.arange(1, 21), values, c=success)
axs[0].set(yscale='log', ylim = [0.000001,1])
axs[1].scatter(np.arange(1, 21), evals, c=success)
axs[1].set(yscale='log')
fig.savefig(name + '.pdf')
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