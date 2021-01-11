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

def paint_complex(function, ansatz, ax_real, ax_imag, df, list_layers):
    df_ = df[(df['function'] == function) & (df['ansatz'] == ansatz)]
    from classes.ApproximantNN import Approximant_complex as App

    if ansatz == 'Weighted':
        from classes.ApproximantNN import NN_complex as cl_function
    else:
        from classes.ApproximantNN import classical_complex_Fourier as cl_function

    function_ = function.split('_')
    real = globals()[f"{function_[0]}"]
    imag = globals()[f"{function_[1]}"]

    for layers in list_layers:
        file_exp = 'results/experiments/' + ansatz + '/' + function + '/%s_layers/result.txt'
        x, y = np.loadtxt(file_exp)

        ax_real.scatter(x, np.real(y), color=colors_classical[str(layers)], label='Experiment %s layers' % layers, zorder=layers)
        ax_imag.scatter(x, np.imag(y), color=colors_classical[str(layers)], label='Experiment %s layers' % layers, zorder=layers)

    C = App(layers, x, ansatz, real, imag)
    ax_real.plot(C.domain, np.real(C.target), color='black')
    ax_imag.plot(C.domain, np.imag(C.target), color='black')


def paint_real(function, ansatz, ax, df, list_layers):
    from classes.ApproximantNN import Approximant_real as App

    func = globals()[f"{function}"]
    for layers in list_layers:
        file_exp = 'results/experiments/' + ansatz + '/' + function + '/%s_layers/result.txt'
        x, y = np.loadtxt(file_exp)
        C = App(layers, x, ansatz, func)

        ax.scatter(x, y, color=colors_classical[str(layers)], label='Classical %s layers' % layers, zorder=layers, s=25)

    ax.plot(C.domain, C.target, color='black', linewidth=3)


def paint_real_2D(function, ansatz, ax, df, layers):
    df_ = df[(df['function'] == function) & (df['ansatz'] == ansatz)].copy()
    from classes.ApproximantNN import Approximant_real_2D as App
    if ansatz == 'Weighted_2D':
        from classes.ApproximantNN import NN_real_2D as cl_function
    else:
        raise NameError('Ansatz not included in 2D')

    func = globals()[f"{function}"]

    file_exp = 'results/experiments/' + ansatz + '/' + function + '/%s_layers/result.txt'
    x, y = np.loadtxt(file_exp)
    C = App(layers, x, func, ansatz)

    ax.scatter(x[:, 0], x[:, 1], y, color=colors_classical[str(layers)], label='Classical %s layers' % layers, zorder=layers, s=25)

    cmap = plt.get_cmap('Greys')
    ax.plot_trisurf(C.domain[:, 0], C.domain[:, 1], C.target-0.1, cmap=cmap, vmin=-4, vmax=1, alpha=0.75)


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


fig = plt.figure(figsize=(15,14))
i = 1
layers=6
ansatz = 'Weighted_2D'
for function in ['Himmelblau','Brent','Adjiman','Threehump']:
    ax = fig.add_subplot(2, 2, i, projection='3d')

    paint_real_2D(function.lower(), ansatz, ax, df, layers)
    pos = ax.get_position()
    pos.x0 -= 0.05
    pos.y1 += 0.05
    if i > 2:
        pos.y1 -= 0.02
        pos.y0 -= 0.02
    ax.set_position(pos)
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.set_zlabel('z', fontsize=20)
    ax.set_title(function, fontsize=20)
    ax.set_xticks([-5,0,5])
    ax.set_yticks([-5, 0, 5])
    ax.set_zticks([-1,0,1])
    ax.tick_params(axis='both', which='major', labelsize=18)
    i += 1

handles = []

handles.append(mlines.Line2D([], [], color=colors_classical[str(layers)], markersize=10, label='Classical' , linewidth=0, marker='o'))
handles.append(mlines.Line2D([], [], color=colors_quantum[str(layers)], markersize=10, label='Quantum' ,
                             linewidth=0, marker='o'))

fig.legend(handles = handles, bbox_to_anchor=(0.48, -0.128, 0.15, .2),borderaxespad=0., mode='expand', fontsize=20, ncol=1)

fig.text(0.35, 0.04, '%s layers'%layers, fontsize=20, fontweight='bold')
fig.savefig('all_2D.pdf')

