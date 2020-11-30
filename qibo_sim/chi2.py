import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('summary.csv')
print(df.head())
color={'Weighted': 'C1', 'Fourier':'C0'}
line={'False':'--', 'True':':'}
marker = {'False':'o', 'True':'P'}

L = 6
fig, axs = plt.subplots(nrows=4, sharex=True, sharey=True, figsize=(8, 22))
for function, ax in zip(['relu', 'step', 'poly', 'tanh'], axs.flatten()):
    for ansatz in ['Weighted', 'Fourier']:
        for quantum in [True, False]:
            chi = []
            for layers in range(1, L + 1):
                chi_ = df[(df['function'] == function) & (df['ansatz'] == ansatz) & (df['quantum'] == quantum) & (df['layers'] == layers)]['chi2'].min()
                chi.append(min(float(chi_), 1))
            ax.plot(list(range(1, L + 1)), chi, color=color[ansatz], linestyle=line[str(quantum)], marker=marker[str(quantum)])

    ax.set(yscale='log')
fig.savefig('real_valued.pdf')

