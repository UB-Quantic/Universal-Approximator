import os
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import matplotlib.lines as mlines

function = 'relu'
ansatz = 'Fourier'
L = 6

colors_quantum = {'1':'#FFB3B3',
                  '2': '#FF8080',
                  '3': '#FF4D4D',
                  '4': '#FF1919',
                  '5': '#CC0000',
                  '6': '#800000'}

colors_classical = {'1':'#CCD4FF',
                  '2': '#99AAFF',
                  '3': '#667FFF',
                  '4': '#3355FF',
                  '5': '#002AFF',
                  '6': '#001EB3'}

df = pd.read_csv('summary.csv')

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

def paint_complex(function, ansatz, ax_real, ax_imag, df):
    df_ = df[(df['function'] == function) & (df['ansatz'] == ansatz)]
    from classes.ApproximantNN import Approximant_complex as App

    if ansatz == 'Weighted':
        from classes.ApproximantNN import NN_complex as cl_function
    else:
        from classes.ApproximantNN import classical_complex_Fourier as cl_function

    function_ = function.split('_')
    real = globals()[f"{function_[0]}"]
    imag = globals()[f"{function_[1]}"]

    for layers in range(1, L + 1):

        df_c = df_[(df_['layers'] == layers) & (df_['quantum'] == False)]
        k_c = df_c['chi2'].idxmin()
        file_c = 'results/classical/' + ansatz + '/' + function + '/%s_layers/%s/result.pkl' % (
        layers, df_c.loc[k_c]['seed'])
        x = np.loadtxt(
            'results/classical/' + ansatz + '/' + function + '/%s_layers/%s/domain.txt' % (layers, df_c.loc[k_c]['trial']))
        C = App(layers, x, ansatz, real, imag)

        with open(file_c, 'rb') as f:
            data_c = pickle.load(f)
        params_c = data_c['x']

        if ansatz == 'Weighted':
            outcomes = cl_function(params_c, layers, x, C.target)
        else:
            outcomes = cl_function(layers, x, C.target)[0]

        ax_real.scatter(C.domain, np.real(outcomes), color=colors_classical[str(layers)], label='Classical %s layers' % layers, zorder=layers)
        ax_imag.scatter(C.domain, np.imag(outcomes), color=colors_classical[str(layers)],
                      label='Classical %s layers' % layers, zorder=layers)

    for layers in range(1, L + 1):
        df_q = df_[(df_['layers']==layers) & (df_['quantum']==True)]
        k_q = df_q['chi2'].idxmin()
        file_q = 'results/quantum/' + ansatz + '/' + function + '/%s_layers/%s/result.pkl'%(layers, df_q.loc[k_q]['trial'])
        x = np.loadtxt('results/quantum/' + ansatz + '/' + function + '/%s_layers/%s/domain.txt'%(layers, df_q.loc[k_q]['trial']))
        with open(file_q, 'rb') as f:
            data_q = pickle.load(f)
        params_q = data_q['x']

        C = App(layers, x, ansatz, real, imag)
        C.set_parameters(params_q)
        r_outcomes = np.zeros_like(C.domain)
        i_outcomes = np.zeros_like(C.domain)
        for j, x in enumerate(C.domain):
            state = C.get_state(x)
            r_outcomes[j] = C.H[0].expectation(state)
            i_outcomes[j] = C.H[1].expectation(state)

        ax_real.scatter(C.domain, r_outcomes, color=colors_quantum[str(layers)], label='Quantum %s layers'%layers, zorder=layers)
        ax_imag.scatter(C.domain, i_outcomes, color=colors_quantum[str(layers)], label='Quantum %s layers' % layers,
                   zorder=layers)

    x_ = np.linspace(np.min(x), np.max(x), 10000)

    ax_real.plot(C.domain, np.real(C.target), color='black')
    ax_imag.plot(C.domain, np.imag(C.target), color='black')


def paint_real(function, ansatz, ax, df):
    df_ = df[(df['function'] == function) & (df['ansatz'] == ansatz)].copy()
    from classes.ApproximantNN import Approximant_real as App
    if ansatz == 'Weighted':
        from classes.ApproximantNN import NN_real as cl_function
    else:
        from classes.ApproximantNN import classical_real_Fourier as cl_function

    func = globals()[f"{function}"]
    for layers in range(1, L + 1):
        df_c = df_[(df_['layers'] == layers) & (df_['quantum'] == False)]
        k_c = df_c['chi2'].idxmin()
        file_c = 'results/classical/' + ansatz + '/' + function + '/%s_layers/%s/result.pkl' % (
        layers, df_c.loc[k_c]['seed'])
        x = np.loadtxt(
            'results/classical/' + ansatz + '/' + function + '/%s_layers/%s/domain.txt' % (layers, df_c.loc[k_c]['trial']))
        C = App(layers, x, ansatz, func)

        with open(file_c, 'rb') as f:
            data_c = pickle.load(f)
        params_c = data_c['x']

        if ansatz == 'Weighted':
            outcomes = cl_function(params_c, layers, x, C.target)
        else:
            outcomes = cl_function(layers, x, C.target)[0]

        ax.scatter(C.domain, outcomes, color=colors_classical[str(layers)], label='Classical %s layers' % layers, zorder=layers, s=20)
    for layers in range(1, L + 1):
        df_q = df_[(df_['layers']==layers) & (df_['quantum']==True)]
        k_q = df_q['chi2'].idxmin()
        file_q = 'results/quantum/' + ansatz + '/' + function + '/%s_layers/%s/result.pkl'%(layers, df_q.loc[k_q]['trial'])
        x = np.loadtxt('results/quantum/' + ansatz + '/' + function + '/%s_layers/%s/domain.txt'%(layers, df_q.loc[k_q]['trial']))
        with open(file_q, 'rb') as f:
            data_q = pickle.load(f)
        params_q = data_q['x']

        C = App(layers, x, ansatz, func)
        C.set_parameters(params_q)
        outcomes = np.zeros_like(C.domain)
        for j, x in enumerate(C.domain):
            state = C.get_state(x)
            outcomes[j] = C.H.expectation(state)

        ax.scatter(C.domain, outcomes, color=colors_quantum[str(layers)], label='Quantum %s layers'%layers, zorder=layers, s=20)

    ax.plot(C.domain, C.target, color='black', linewidth=3)


fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(24,12), sharex=True, sharey=True)


i = 0
for ansatz in ['Fourier','Weighted']:
    for function in ['tanh', 'step', 'poly', 'relu']:
        ax = axs.flatten()[i]
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=18)
        paint_real(function, ansatz, ax, df)
        pos = ax.get_position()
        pos.x0 -= 0.05
        pos.x1 -= 0.05
        ax.set_position(pos)

        i+=1



axs.flatten()[0].set_ylabel(r'$f(x)$', fontsize=24)
axs.flatten()[0].set_title(r'$\tanh(5 x)$', fontsize=24)
axs.flatten()[1].set_title(r'${\rm step}(x)$', fontsize=24)
axs.flatten()[2].set_title(r'${\rm poly}(x)$', fontsize=24)
axs.flatten()[3].set_title(r'${\rm ReLU}(x)$', fontsize=24)
axs.flatten()[4].set_ylabel(r'$f(x)$', fontsize=24)
axs.flatten()[4].set_xlabel(r'$x$', fontsize=24)
axs.flatten()[5].set_xlabel(r'$x$', fontsize=24)
axs.flatten()[5].set_xlabel(r'$x$', fontsize=24)
axs.flatten()[6].set_xlabel(r'$x$', fontsize=24)
axs.flatten()[7].set_xlabel(r'$x$', fontsize=24)

fig.text(0.005, 0.65, 'Fourier', rotation='vertical', fontsize=30, fontweight='bold')
fig.text(0.005, 0.26, 'UAT', rotation='vertical', fontsize=30, fontweight='bold')

handles = []
for layers in range(1, L+1):
    handles.append(mlines.Line2D([], [], color=colors_quantum[str(layers)], markersize=10, label= '%s layers'%layers, linewidth=0, marker='o'))

fig.text(0.88, 0.82, 'Quantum', fontsize=22, fontweight='bold')
fig.legend(handles = handles, bbox_to_anchor=(0.88, 0.6, 0.1, .4), loc='lower left', borderaxespad=0., mode='expand', fontsize=20)

handles = []
for layers in range(1, L + 1):
    handles.append(mlines.Line2D([], [], color=colors_classical[str(layers)], markersize=10, label='%s layers' % layers,  linewidth=0, marker='o'))

fig.text(0.88, 0.36, 'Classical', fontsize=22, fontweight='bold')
fig.legend(handles = handles, bbox_to_anchor=(0.88, 0.14, 0.1, .4), loc='lower left', borderaxespad=0., mode='expand', fontsize=20)

handles = []
handles.append(mlines.Line2D([], [], color='black', markersize=0, label='Target', linewidth=4))
fig.legend(handles = handles, bbox_to_anchor=(0.88, 0.48, 0.1, .4), loc='lower left', borderaxespad=0., mode='expand', fontsize=20)

fig.savefig('all_reals.pdf')

