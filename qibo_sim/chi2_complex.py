import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

functions = ['relu', 'step', 'poly', 'tanh']

def paint_matrix(df):
    matrix = np.array((len(functions), len(functions)))
    for i, f1 in enumerate(functions):
        for j, f2 in enumerate(functions):
            c_function = f1 + '_' + f2
    pass

df = pd.read_csv('summary.csv')

color={'Weighted': 'C1', 'Fourier':'C0'}
line={'False':'--', 'True':'..'}
marker = {'False':'o', 'True':'P'}

L = 6
fig, axs = plt.subplots(nrows=4)
for function, ax in zip(functions, axs.flatten()):
    for ansatz in ['Weighted', 'Fourier']:
        for quantum in ['True', 'False']:
            chi = []
            for layers in range(1, L + 1):
                chi_ = df[(df['function'] == function) &
                               (df['ansatz'] == ansatz) &
                               (df['Quantum'] == quantum) &
                               (df['layers'] == layers)]['chi2'].min()
                chi.append(min(chi_, 1))
            ax.plot(list(range(1, L + 1)), chi, color=color[ansatz], linestyle=line[quantum], marker=marker[quantum])

fig.savefig('real_valued.pdf')

