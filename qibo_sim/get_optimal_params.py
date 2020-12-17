import pickle
import pandas as pd

ansatz = 'Weighted'
function = 'poly'
quantum = True
layers=6

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

print(data['x'])