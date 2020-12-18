import os
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import matplotlib.lines as mlines

function = 'tanh_relu'
layers = 6

colors = {'Fourier':'orange',
          'UAT':'maroon'}

marker = {'UAT':'X', 'Fourier':'^'}

ansatze={'UAT':'Weighted', 'Fourier':'Fourier'}


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

def paint_complex(function, ax_real, ax_imag, df, layers=6):

    from classes.ApproximantNN import Approximant_complex as App
    function_ = function.split('_')
    real = globals()[f"{function_[0]}"]
    imag = globals()[f"{function_[1]}"]

    ansatz = 'UAT'
    from classes.ApproximantNN import NN_complex as cl_function
    df_ = df[(df['function'] == function) & (df['quantum'] == True)]

    df_w = df_[(df_['layers'] == layers) & (df_['ansatz'] == ansatze[ansatz])]
    k_w = df_w['chi2'].idxmin()
    file_w = 'results/quantum/' + ansatze[ansatz] + '/' + function + '/%s_layers/%s/result.pkl' % (
    layers, df_w.loc[k_w]['trial'])
    x = np.loadtxt(
        'results/classical/' + ansatze[ansatz] + '/' + function + '/%s_layers/%s/domain.txt' % (layers, df_w.loc[k_w]['trial']))
    C = App(layers, x, ansatze[ansatz], real, imag)

    with open(file_w, 'rb') as f:
        data_w = pickle.load(f)
    params_w = data_w['x']
    C.set_parameters(params_w)
    r_outcomes = np.zeros_like(C.domain)
    i_outcomes = np.zeros_like(C.domain)
    for j, x in enumerate(C.domain):
        state = C.get_state(x)
        r_outcomes[j] = C.H[0].expectation(state)
        i_outcomes[j] = C.H[1].expectation(state)
    ax_real.plot(C.domain, r_outcomes, color=colors[ansatz], marker=marker[ansatz],
                  markersize=10, linewidth=2, linestyle='--')
    ax_imag.plot(C.domain, i_outcomes, color=colors[ansatz], marker=marker[ansatz],
                  markersize=10, linewidth=2, linestyle='--')

    ansatz = 'Fourier'
    df_f = df_[(df_['layers'] == layers) & (df_['ansatz'] == ansatze[ansatz])]
    k_f = df_f['chi2'].idxmin()
    file_f = 'results/quantum/' + ansatze[ansatz] + '/' + function + '/%s_layers/%s/result.pkl' % (
        layers, df_f.loc[k_f]['trial'])
    x = np.loadtxt(
        'results/classical/' + ansatz + '/' + function + '/%s_layers/%s/domain.txt' % (layers, df_f.loc[k_f]['trial']))
    C = App(layers, x, ansatze[ansatz], real, imag)

    with open(file_f, 'rb') as f:
        data_f = pickle.load(f)
    params_f = data_f['x']
    C.set_parameters(params_f)
    r_outcomes = np.zeros_like(C.domain)
    i_outcomes = np.zeros_like(C.domain)
    for j, x in enumerate(C.domain):
        state = C.get_state(x)
        r_outcomes[j] = C.H[0].expectation(state)
        i_outcomes[j] = C.H[1].expectation(state)

    ax_real.plot(C.domain, r_outcomes, color=colors[ansatz], marker=marker[ansatz],
                  markersize=10, linewidth=2, linestyle='--')
    ax_imag.plot(C.domain, i_outcomes, color=colors[ansatz], marker=marker[ansatz],
                  markersize=10, linewidth=2, linestyle='--')


    ax_real.plot(C.domain, np.real(C.target), color='black', zorder=10)
    ax_imag.plot(C.domain, np.imag(C.target), color='black', zorder=10)


def paint_real(function, ax, df, layers=6):
    from classes.ApproximantNN import Approximant_real as App
    func = globals()[f"{function}"]

    ansatz = 'UAT'
    from classes.ApproximantNN import NN_complex as cl_function
    df_ = df[(df['function'] == function) & (df['quantum'] == True)]

    df_w = df_[(df_['layers'] == layers) & (df_['ansatz'] == ansatze[ansatz])]
    k_w = df_w['chi2'].idxmin()
    file_w = 'results/quantum/' + ansatze[ansatz] + '/' + function + '/%s_layers/%s/result.pkl' % (
        layers, df_w.loc[k_w]['trial'])
    x = np.loadtxt(
        'results/classical/' + ansatze[ansatz] + '/' + function + '/%s_layers/%s/domain.txt' % (
        layers, df_w.loc[k_w]['trial']))
    C = App(layers, x, ansatze[ansatz], func)

    with open(file_w, 'rb') as f:
        data_w = pickle.load(f)
    params_w = data_w['x']
    C.set_parameters(params_w)
    outcomes = np.zeros_like(C.domain)
    for j, x in enumerate(C.domain):
        state = C.get_state(x)
        outcomes[j] = C.H.expectation(state)

    ax.plot(C.domain, outcomes, color=colors[ansatz], marker=marker[ansatz],
                 markersize=10, linewidth=2, linestyle='--')

    ansatz = 'Fourier'
    df_f = df_[(df_['layers'] == layers) & (df_['ansatz'] == ansatze[ansatz])]
    k_f = df_f['chi2'].idxmin()
    file_f = 'results/quantum/' + ansatze[ansatz] + '/' + function + '/%s_layers/%s/result.pkl' % (
        layers, df_f.loc[k_f]['trial'])
    x = np.loadtxt(
        'results/classical/' + ansatz + '/' + function + '/%s_layers/%s/domain.txt' % (layers, df_f.loc[k_f]['trial']))
    C = App(layers, x, ansatze[ansatz], func)

    with open(file_f, 'rb') as f:
        data_f = pickle.load(f)
    params_f = data_f['x']
    C.set_parameters(params_f)
    outcomes = np.zeros_like(C.domain)
    for j, x in enumerate(C.domain):
        state = C.get_state(x)
        outcomes[j] = C.H.expectation(state)

    ax.plot(C.domain, outcomes, color=colors[ansatz], marker=marker[ansatz],
                 markersize=10, linewidth=2, linestyle='--')

    ax.plot(C.domain, C.target, color='black', zorder=10)

def paint_real_2D(function, ax, df, layers=6):
    from classes.ApproximantNN import Approximant_real_2D as App
    func = globals()[f"{function}"]

    ansatz = 'UAT'
    df_ = df[(df['function'] == function) & (df['quantum'] == True)]

    df_w = df_[(df_['layers'] == layers) & (df_['ansatz'] == ansatze[ansatz])]
    k_w = df_w['chi2'].idxmin()
    file_w = 'results/quantum/' + ansatze[ansatz] + '/' + function + '/%s_layers/%s/result.pkl' % (
        layers, df_w.loc[k_w]['trial'])
    x = np.loadtxt(
        'results/classical/' + ansatze[ansatz] + '/' + function + '/%s_layers/%s/domain.txt' % (
        layers, df_w.loc[k_w]['trial']))
    C = App(layers, x, func, ansatze[ansatz])

    with open(file_w, 'rb') as f:
        data_w = pickle.load(f)
    params_w = data_w['x']
    C.set_parameters(params_w)
    outcomes = np.zeros(len(C.domain))
    for j, x in enumerate(C.domain):
        state = C.get_state(x)
        outcomes[j] = C.H.expectation(state)

    ax.scatter(C.domain[:, 0], C.domain[:, 1], outcomes, color=colors[ansatz], marker=marker[ansatz],s=10)

    '''ansatz = 'Fourier'
    df_f = df_[(df_['layers'] == layers) & (df_['ansatz'] == ansatze[ansatz])]
    k_f = df_f['chi2'].idxmin()
    file_f = 'results/quantum/' + ansatze[ansatz] + '/' + function + '/%s_layers/%s/result.pkl' % (
        layers, df_f.loc[k_f]['trial'])
    x = np.loadtxt(
        'results/classical/' + ansatz + '/' + function + '/%s_layers/%s/domain.txt' % (layers, df_f.loc[k_f]['trial']))
    C = App(layers, x, ansatze[ansatz], func)

    with open(file_f, 'rb') as f:
        data_f = pickle.load(f)
    params_f = data_f['x']
    C.set_parameters(params_f)
    outcomes = np.zeros_like(C.domain)
    for j, x in enumerate(C.domain):
        state = C.get_state(x)
        outcomes[j] = C.H.expectation(state)

    ax.plot(C.domain, outcomes, color=colors[ansatz], marker=marker[ansatz],
                 markersize=10, linewidth=2, linestyle='--')'''

    ax.plot_trisurf(C.domain[:, 0], C.domain[:, 1], C.target, color='black', alpha=0.5)

fig, axs = plt.subplots(nrows=2)
paint_complex('tanh_relu', axs[0], axs[1], df)
plt.show()

fig, axs = plt.subplots()
paint_real('relu', axs, df)
plt.show()

ansatze={'UAT':'Weighted_2D', 'Fourier':'Fourier_2D'}
fig = plt.figure()
ax = fig.gca(projection='3d')
paint_real_2D('himmelblau', ax, df)
ax.set_xlabel('x', fontsize=20)
ax.set_ylabel('y', fontsize=20)
ax.set_zlabel('Himmelblau(x, y)', fontsize=20, rotation=180)
plt.show()

'''fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(24,12), sharex=True, sharey=True)


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



fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,12), sharex=True, sharey=True)


i = 0
function = 'tanh_relu'
for ansatz in ['Fourier','Weighted']:
    ax = axs[i]
    print(ax.shape)
    ax[0].grid(True)
    ax[0].tick_params(axis='both', which='major', labelsize=18)
    ax[1].grid(True)
    ax[1].tick_params(axis='both', which='major', labelsize=18)
    paint_complex(function, ansatz, ax[0], ax[1], df)
    i += 1
    pos = ax[1].get_position()
    pos.x0 -= 0.05
    pos.x1 -= 0.05
    ax[1].set_position(pos)




axs.flatten()[0].set_ylabel(r'$f(x)$', fontsize=24)
axs.flatten()[2].set_ylabel(r'$f(x)$', fontsize=24)

fig.text(0.005, 0.65, 'Fourier', rotation='vertical', fontsize=30, fontweight='bold')
fig.text(0.005, 0.26, 'UAT', rotation='vertical', fontsize=30, fontweight='bold')

fig.text(0.15, 0.9, r'Real part = $\tanh(5x)$', fontsize=30)
fig.text(0.52, 0.9, r'Imag part = ${\rm ReLU}(x)$', fontsize=30)

handles = []
for layers in range(1, L+1):
    handles.append(mlines.Line2D([], [], color=colors_quantum[str(layers)], markersize=10, label= '%s layers'%layers, linewidth=0, marker='o'))

fig.text(0.86, 0.82, 'Quantum', fontsize=22, fontweight='bold')
fig.legend(handles = handles, bbox_to_anchor=(0.86, 0.6, 0.138, .4), loc='lower left', borderaxespad=0., mode='expand', fontsize=20)

handles = []
for layers in range(1, L + 1):
    handles.append(mlines.Line2D([], [], color=colors_classical[str(layers)], markersize=10, label='%s layers' % layers,  linewidth=0, marker='o'))

fig.text(0.86, 0.36, 'Classical', fontsize=22, fontweight='bold')
fig.legend(handles = handles, bbox_to_anchor=(0.86, 0.14, 0.138, .4), loc='lower left', borderaxespad=0., mode='expand', fontsize=20)

handles = []
handles.append(mlines.Line2D([], [], color='black', markersize=0, label='Target', linewidth=4))
fig.legend(handles = handles, bbox_to_anchor=(0.86, 0.48, 0.138, .4), loc='lower left', borderaxespad=0., mode='expand', fontsize=20)

fig.savefig(function + '_complex.pdf')
'''
