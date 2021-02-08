import pickle
import pandas as pd

ansatz = 'Weighted'
function = 'relu'
quantum = True
layers=4

df = pd.read_csv('summary.csv')

df = df[(df['ansatz']==ansatz) & (df['function']==function) & (df['layers']==layers) & (df['quantum']==quantum)]

id = df['chi2'].idxmin()

if quantum:
    q = 'quantum'
else:
    q = 'classical'
file_name = 'results/' + q + '/' + ansatz + '/' + function + '/' + '%s_layers'%layers + '/' + str(df['trial'].loc[[id]].values[0]) + '/result.pkl'

with open(file_name, 'rb') as f:
    data = pickle.load(f)

parameters = data['x']

print(parameters)
import numpy as np
for x in [-.5, 0, 1]:
    angles = np.zeros(2*layers)
    i = 0
    j = 0
    for l in range(1, layers + 1):
        angles[i] = x * parameters[j + 1] + parameters[j]
        angles[i + 1] = parameters[j + 2]
        i += 2
        j += 3

    print('__'*50)
    for a in angles:
        print('& \multicolumn{2}{c}{%.3f}'%(a%(2*np.pi)))
