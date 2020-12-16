import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('summary.csv')
color={'True': 'red', 'False':'blue'}
line={'False':'--', 'True':':'}
marker = {'UAT':'X', 'Fourier':'^'}

functions = ['tanh', 'step', 'poly', 'relu']

ansatze={'UAT':'Weighted', 'Fourier':'Fourier'}

titles = {'tanh': r'f(x) = $\tanh(5 x)$','poly':r'$f(x) = {\rm poly}(x)$', 'step': r'$f(x) = {\rm step}(x)$',  'relu':r'$f(x) = {\rm ReLU}(x)$'}

L = 6
fig, axs = plt.subplots(nrows=4, sharex=True, sharey=False, figsize=(9, 18))
handles = []
i = 0
for function, ax in zip(functions, axs.flatten()):
    for ansatz in ['UAT', 'Fourier']:
        for quantum in [True, False]:
            chi = []
            for layers in range(1, L + 1):
                chi_ = df[(df['function'] == function) & (df['ansatz'] == ansatze[ansatz]) & (df['quantum'] == quantum) & (df['layers'] == layers)]['chi2'].min()
                chi.append(min(float(chi_), 1))
            ax.plot(list(range(1, L + 1)), chi, color=color[str(quantum)],
                    linestyle=line[str(quantum)],
                    marker=marker[ansatz], markersize=15)
    pos = ax.get_position()
    pos.x0 += 0.02
    pos.x1 += 0.02
    pos.y1 = 1 - 0.025 - i * 0.2250
    pos.y0 = pos.y1 - 0.19
    i += 1
    ax.set_position(pos)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_title(titles[function], fontsize=22)
    ax.set(yscale='log')
    ax.set_ylabel(ylabel=r'$\chi^2$', fontsize=24)
    ax.grid(True)

ax.set_xlabel('Layers', fontsize=24)

import matplotlib.lines as mlines
for quantum in [False, True]:
    for ansatz in ['Fourier', 'UAT']:
        if quantum:
            q = 'Quantum'
        else:
            q = 'Classical'
        handles.append(mlines.Line2D([], [], color=color[str(quantum)],
                    linestyle=line[str(quantum)],
                    marker=marker[ansatz],
                          markersize=10, label= q + ' ' + ansatz))

fig.legend(handles = handles, bbox_to_anchor=(0.18, 0.01, 0.75, .4), loc='lower left', borderaxespad=0., mode='expand',
           ncol=2, prop={'size': 20})
fig.savefig('real_valued.pdf')


L = 6
fig, axs = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=False, figsize=(20, 14))
handles = []

titles = {'tanh': r'$\tanh(5 x)$','poly':r'${\rm poly}(x)$', 'step': r'${\rm step}(x)$',  'relu':r'${\rm ReLU}(x)$'}

i = 0
for i, function1 in enumerate(functions):
    fig.text(.02, 1 - i * 0.85 / 4 - .2, r'$\mathbf{X} = $' + titles[function1], rotation='vertical', fontsize=24)
    for j, function2 in enumerate(functions):
        if i == 0:
            fig.text(0.15 + j * 0.92 / 4, 0.98, r'$\mathbf{Y} = $' + titles[function2],
                     fontsize=24)
        function = function1 + '_' + function2
        ax = axs.flatten()[4 * i + j]
        for ansatz in ['UAT', 'Fourier']:
            for quantum in [True, False]:
                chi = []
                for layers in range(1, L + 1):
                    chi_ = df[(df['function'] == function) & (df['ansatz'] == ansatze[ansatz]) & (df['quantum'] == quantum) & (df['layers'] == layers)]['chi2'].min()
                    chi.append(min(float(chi_), 1))
                ax.plot(list(range(1, L + 1)), chi, color=color[str(quantum)],
                    linestyle=line[str(quantum)],
                    marker=marker[ansatz], markersize=10)
                ax.grid(True)

        function_ = function.split('_')
        #tit = titles[function_[0]] + r'$\; + \; i \,$' + titles[function_[1]]
        ax.set(yscale='log')
        pos = ax.get_position()
        pos.x0 = 0.1 + j * 0.92 / 4
        pos.x1 = pos.x0 + 0.19
        pos.y0 = 0.12 + 0.85 / 4 * (3 - i)
        pos.y1 = pos.y0 + 0.2
        ax.set_position(pos)
        ax.tick_params(axis='both', which='major', labelsize=18)
        #ax.set_title(tit, fontsize=10)
        ax.set(yscale='log')
        if j == 0:
            ax.set_ylabel(ylabel=r'$\chi^2$', fontsize=24)
            ax.tick_params(axis='both', which='major', labelsize=15)
        else:
            ax.tick_params(axis='both', which='major', labelsize=15)
        ax.grid(True)

axs.flatten()[12].set_xlabel('Layers', fontsize=20)
axs.flatten()[13].set_xlabel('Layers', fontsize=20)
axs.flatten()[15].set_xlabel('Layers', fontsize=20)
axs.flatten()[14].set_xlabel('Layers', fontsize=20)

axs.flatten()[0].set_ylabel(r'$\chi^2$', fontsize=20)
axs.flatten()[4].set_ylabel(r'$\chi^2$', fontsize=20)  # ylabel=r'$\chi^2$',
axs.flatten()[8].set_ylabel(r'$\chi^2$', fontsize=20)
axs.flatten()[12].set_ylabel(r'$\chi^2$', fontsize=20)


import matplotlib.lines as mlines
for quantum in [False, True]:
    for ansatz in ['Fourier', 'UAT']:
        if quantum:
            q = 'quantum'
        else:
            q = 'classical'
        handles.append(mlines.Line2D([], [], color=color[str(quantum)],
                    linestyle=line[str(quantum)],
                    marker=marker[ansatz],
                          markersize=10, label= q + ' ' + ansatz))

fig.legend(handles = handles, bbox_to_anchor=(0.25, 0.002, 0.4, .4), loc='lower left', borderaxespad=0., mode='expand',
           ncol=2, prop={'size': 20})
fig.savefig('complex_valued.pdf')

