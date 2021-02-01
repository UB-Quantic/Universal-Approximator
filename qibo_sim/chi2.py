import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('summary.csv')
color={'True': 'tab:orange', 'False':'tab:blue', 'Experiment':'tab:purple'}
line={'False':'--', 'True':':', 'Experiment':'-.'}
marker = {'UAT':'X', 'Fourier':'^', 'Experiment':'+'}

functions = ['tanh', 'step', 'poly', 'relu']
#functions = ['tanh', 'relu']
ansatze={'UAT':'Weighted', 'Fourier':'Fourier'}

titles = {'tanh': r'f(x) = $\tanh(5 x)$','poly':r'$f(x) = {\rm poly}(x)$', 'step': r'$f(x) = {\rm step}(x)$',  'relu':r'$f(x) = {\rm ReLU}(x)$'}


def step(x):
    step.name = 'step'
    return 0.5 * (np.sign(x) + 1)

def cosine(x):
    cosine.name = 'cosine'
    return np.cos(2*np.pi*x)

def sigmoid(x, a=10):
    sigmoid.name = 'sigmoid'
    return 1 / (1 + np.exp(-a * x))

def tanh(x, a=5):
    tanh.name = 'tanh'
    return np.tanh(a * x)


def angulator(x):
    return 2 * np.pi * x

def zero(x):
    tanh.name = 'zero'
    #return 0.5 * (np.tanh(x) + 1)
    return 0

def tanh_2(x):
    tanh.name = 'tanh'
    #return 0.5 * (np.tanh(x) + 1)
    return (np.tanh(np.linalg.norm(x))**2)

def relu(x):
    relu.name = 'relu'
    return x * (x > 0)

def poly(x):
    poly.name= 'poly'
    return np.abs(3*x**3 * (1 - x**4))

def himmelblau(x):
    himmelblau.name = 'himmelblau'
    for x_ in x:
        yield (x_[0]**2 + x_[1] - 11)**2 + (x_[0] + x_[1]**2 - 7)**2

def brent(x):
    brent.name = 'brent'
    for x_ in x:
        yield np.linalg.norm(x_ / 2)**2 + np.exp(-np.linalg.norm(x_/2 - 5)**2)

def threehump(x):
    threehump.name = 'threehump'
    for x_ in x * 2 / 5:
        yield 2 * x_[0]**2 - 1.05 * x_[0]**4 + 1/6 * x_[0]**6 + x_[0] * x_[1] + x_[1]**2

def adjiman(x):
    adjiman.name = 'adjiman'
    for x_ in x:
        yield np.cos(x_[0]) * np.sin(x_[1]) - x_[0] / (x_[1]**2 + 1)

def rosenbrock(x):
    rosenbrock.name = 'rosenbrock'
    for x_ in x:
        yield (1 - .4*x_[0])**2 + 100 * (x_[1] - (.4*x_[0])**2)**2

plt.style.use('seaborn')

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

        quantum = 'Experiment'
        chi = []
        for layers in range(1, L + 1):
            file_exp = 'results/experiment/' + ansatze[ansatz] + '/' + function + '/%s_layers/results.txt' % layers
            x, y = np.loadtxt(file_exp)
            y = 2*y - 1
            from classes.ApproximantNN import Approximant_real as App

            func = globals()[f"{function}"]
            C = App(layers, x, ansatze[ansatz], func)
            chi_ = np.mean(np.abs(y - C.target) ** 2)
            chi.append(min(float(chi_), 1))
        ax.plot(list(range(1, L + 1)), chi, color=color[str(quantum)],
                linestyle=line[str(quantum)],
                marker=marker[ansatz], markersize=15)
        ax.grid(True)
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

for ansatz in ['Fourier', 'UAT']:
    for quantum in [False, True]:

        if quantum:
            q = 'Simulation'
        else:
            q = 'Classical'
        handles.append(mlines.Line2D([], [], color=color[str(quantum)],
                    linestyle=line[str(quantum)],
                    marker=marker[ansatz],
                          markersize=10, label= q + ' ' + ansatz))

    quantum='Experiment'
    handles.append(mlines.Line2D([], [], color=color[str(quantum)],
                linestyle=line[str(quantum)],
                marker=marker[ansatz],
                      markersize=10, label= quantum + ' ' + ansatz))

fig.legend(handles = handles, bbox_to_anchor=(0.18, 0.002, 0.75, .4), loc='lower left', borderaxespad=0., mode='expand',
           ncol=2, prop={'size': 18})
fig.savefig('chi_real.pdf')


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

            try:
                quantum = 'Experiment'
                chi = []
                for layers in range(1, L + 1):
                    file_exp_x = 'results/experiment/' + ansatze[ansatz] + '/' + function + '/%s_layers/results_x.txt' % layers
                    domain, x = np.loadtxt(file_exp_x)
                    x = 2 * x - 1

                    file_exp_y = 'results/experiment/' + ansatze[
                        ansatz] + '/' + function + '/%s_layers/results_y.txt' % layers
                    domain, y = np.loadtxt(file_exp_y)
                    y = 2 * y - 1
                    from classes.ApproximantNN import Approximant_real as App

                    func1 = globals()[f"{function1}"]
                    func2 = globals()[f"{function2}"]
                    from classes.ApproximantNN import Approximant_complex as App
                    C = App(layers, domain, ansatze[ansatz], func1, func2)
                    chi_ = np.mean(np.abs(x + 1j * y - C.target) ** 2)
                    chi.append(min(float(chi_), 1))
                ax.plot(list(range(1, L + 1)), chi, color=color[str(quantum)],
                        linestyle=line[str(quantum)],
                        marker=marker[ansatz], markersize=10)
            except:pass

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
ansatz = 'Fourier'
for quantum in [False, True]:

    if quantum:
        q = 'Simulation'
    else:
        q = 'Classical'
    handles.append(mlines.Line2D([], [], color=color[str(quantum)],
                linestyle=line[str(quantum)],
                marker=marker[ansatz],
                      markersize=10, label= q + ' ' + ansatz))

quantum='Experiment'
handles.append(mlines.Line2D([], [], color=color[str(quantum)],
            linestyle=line[str(quantum)],
            marker=marker[ansatz],
                  markersize=10, label= quantum + ' ' + ansatz))

fig.legend(handles = handles, bbox_to_anchor=(0.23, 0.002, 0.165, .4), loc='lower left', borderaxespad=0., mode='expand',
           ncol=1, prop={'size': 18})

ansatz = 'UAT'
handles=[]
for quantum in [False, True]:

    if quantum:
        q = 'Simulation'
    else:
        q = 'Classical'
    handles.append(mlines.Line2D([], [], color=color[str(quantum)],
                linestyle=line[str(quantum)],
                marker=marker[ansatz],
                      markersize=10, label= q + ' ' + ansatz))

quantum='Experiment'
handles.append(mlines.Line2D([], [], color=color[str(quantum)],
            linestyle=line[str(quantum)],
            marker=marker[ansatz],
                  markersize=10, label= quantum + ' ' + ansatz))

fig.legend(handles = handles, bbox_to_anchor=(0.47, 0.002, 0.15, .4), loc='lower left', borderaxespad=0., mode='expand',
           ncol=1, prop={'size': 18})

fig.savefig('chi_complex.pdf')


line={'False':'--', 'True':':', 'Experiment':'-.'}
marker = {'UAT':'X', 'Fourier':'^', 'Experiment':'+'}

functions = ['himmelblau', 'brent', 'threehump', 'adjiman']

ansatze={'UAT':'Weighted_2D', 'Fourier':'Fourier_2D'}

titles = {'adjiman': r'${\rm Adjiman(x, y)}$','threehump':r'${\rm Threehump(x, y)}$', 'brent': r'${\rm Brent(x, y)}$',  'himmelblau':r'${\rm Himmelblau(x, y)}$'}

L = 6
fig, axs = plt.subplots(nrows=4, sharex=True, sharey=False, figsize=(9, 18))
handles = []
i = 0
for function, ax in zip(functions, axs.flatten()):
    for ansatz in ['UAT']:
        for quantum in [True, False]:
            chi = []
            for layers in range(1, L + 1):
                chi_ = df[(df['function'] == function) & (df['ansatz'] == ansatze[ansatz]) & (df['quantum'] == quantum) & (df['layers'] == layers)]['chi2'].min()
                chi.append(min(float(chi_), 1))
            ax.plot(list(range(1, L + 1)), chi, color=color[str(quantum)],
                    linestyle=line[str(quantum)],
                    marker=marker[ansatz], markersize=15)

    try:
        quantum = 'Experiment'
        x_0 = np.linspace(-5, 5, 25)
        x_1 = np.linspace(-5, 5, 25)
        from itertools import product

        x = np.array(list(product(x_0, x_1)))
        chi = []
        for layers in range(1, L + 1):
            file_exp = 'results/experiment/' + ansatze[ansatz] + '/' + function + '/%s_layers/results.txt' % layers
            y = np.loadtxt(file_exp)
            y = y.transpose()
            y = 2 * y.flatten() - 1
            from classes.ApproximantNN import Approximant_real_2D as App

            func = globals()[f"{function}"]
            C = App(layers, x, func, ansatze[ansatz])
            chi_ = np.mean(np.abs(y - C.target) ** 2)
            chi.append(min(float(chi_), 1))
        ax.plot(list(range(1, L + 1)), chi, color=color[str(quantum)],
                linestyle=line[str(quantum)],
                marker=marker[ansatz], markersize=15)
    except:pass
    ax.grid(True)

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

for ansatz in ['UAT']:
    for quantum in [False, True]:
        if quantum:
            q = 'Simulation'
        else:
            q = 'Classical'
        handles.append(mlines.Line2D([], [], color=color[str(quantum)],
                    linestyle=line[str(quantum)],
                    marker=marker[ansatz],
                          markersize=10, label= q + ' ' + ansatz))

    quantum = 'Experiment'
    handles.append(mlines.Line2D([], [], color=color[str(quantum)],
                                 linestyle=line[str(quantum)],
                                 marker=marker[ansatz],
                                 markersize=10, label=quantum + ' ' + ansatz))

fig.legend(handles = handles, bbox_to_anchor=(0.63, 0.01, 0.35, .35), loc='lower left', borderaxespad=0., mode='expand',
           ncol=1, prop={'size': 18})
fig.savefig('chi_2D.pdf')

