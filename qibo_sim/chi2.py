import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('summary.csv')
color={'True': 'red', 'False':'blue'}
line={'False':'--', 'True':':'}
marker = {'Weighted':'$W$', 'Fourier':'$F$'}

functions = ['tanh', 'step', 'poly', 'relu']

titles = {'tanh': r'$\tanh(5 x)$','poly':r'${\rm poly}(x)$', 'step': r'${\rm step}(x)$',  'relu':r'${\rm ReLU}(x)$'}

L = 6
fig, axs = plt.subplots(nrows=4, sharex=True, sharey=True, figsize=(6, 14))
handles = []
for function, ax in zip(functions, axs.flatten()):
    for ansatz in ['Weighted', 'Fourier']:
        for quantum in [True, False]:
            chi = []
            for layers in range(1, L + 1):
                chi_ = df[(df['function'] == function) & (df['ansatz'] == ansatz) & (df['quantum'] == quantum) & (df['layers'] == layers)]['chi2'].min()
                chi.append(min(float(chi_), 1))
            ax.plot(list(range(1, L + 1)), chi, color=color[str(quantum)],
                    linestyle=line[str(quantum)],
                    marker=marker[ansatz])
    ax.set(title=titles[function], yscale='log', ylabel=r'$\chi^2$')
    ax.grid(True)
ax.set(xlabel='Layers')

import matplotlib.lines as mlines
for quantum in [False, True]:
    for ansatz in ['Fourier', 'Weighted']:
        if quantum:
            q = 'quantum'
        else:
            q = 'classical'
        handles.append(mlines.Line2D([], [], color=color[str(quantum)],
                    linestyle=line[str(quantum)],
                    marker=marker[ansatz],
                          markersize=10, label= ansatz + ' ' + q))

fig.legend(handles = handles, bbox_to_anchor=(0.6, 0.01, 0.3, .4), loc='lower left', borderaxespad=0., mode='expand')
fig.savefig('real_valued.png')



import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('summary.csv')

color={'True': 'red', 'False':'blue'}
line={'False':'--', 'True':':'}
marker = {'Weighted':'$W$', 'Fourier':'$F$'}

L = 6
fig, axs = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, figsize=(16, 14))
handles = []
i = 0
for function1 in functions:
    for function2 in functions:
        function = function1 + '_' + function2
        ax = axs.flatten()[i]
        for ansatz in ['Weighted', 'Fourier']:
            for quantum in [True, False]:
                chi = []
                for layers in range(1, L + 1):
                    chi_ = df[(df['function'] == function) & (df['ansatz'] == ansatz) & (df['quantum'] == quantum) & (df['layers'] == layers)]['chi2'].min()
                    chi.append(min(float(chi_), 1))
                ax.plot(list(range(1, L + 1)), chi, color=color[str(quantum)],
                    linestyle=line[str(quantum)],
                    marker=marker[ansatz])
                ax.grid(True)

        function_ = function.split('_')
        tit = titles[function_[0]] + r'$\; + \; i \,$' + titles[function_[1]]
        ax.set(title=tit, yscale='log')
        i += 1

axs.flatten()[12].set(xlabel='Layers')
axs.flatten()[13].set(xlabel='Layers')
axs.flatten()[15].set(xlabel='Layers')
axs.flatten()[14].set(xlabel='Layers')

axs.flatten()[0].set(ylabel=r'$\chi^2$')
axs.flatten()[4].set(ylabel=r'$\chi^2$')  # ylabel=r'$\chi^2$',
axs.flatten()[8].set(ylabel=r'$\chi^2$')
axs.flatten()[12].set(ylabel=r'$\chi^2$')
import matplotlib.lines as mlines
for quantum in [False, True]:
    for ansatz in ['Fourier', 'Weighted']:
        if quantum:
            q = 'quantum'
        else:
            q = 'classical'
        handles.append(mlines.Line2D([], [], color=color[str(quantum)],
                    linestyle=line[str(quantum)],
                    marker=marker[ansatz],
                          markersize=10, label= ansatz + ' ' + q))

fig.legend(handles = handles, bbox_to_anchor=(0.85, 0.01, 0.12, .4), loc='lower left', borderaxespad=0., mode='expand')
fig.savefig('complex_valued.png')

