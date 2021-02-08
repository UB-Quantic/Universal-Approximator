import os
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import matplotlib.lines as mlines

function = 'relu'
ansatz = 'Fourier'
L = 5

list_layers = [2,4,6]

colors = {'quantum': '#CC0000',
          'experiment': '#009933',
          'classical': '#002AFF'}

'''colors = {'quantum': 'tab:blue',
          'experiment': 'tab:purple',
          'classical': 'tab:orange'}
'''
marker = {'classical':'^',
          'quantum':'o',
          'experiment':'s'}

df = pd.read_csv('summary.csv')

error = 0.015

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

def paint_complex(function, ansatz, ax_real, ax_imag, df, layers):
    error = 0.015
    df_ = df[(df['function'] == function) & (df['ansatz'] == ansatz)]
    from classes.ApproximantNN import Approximant_complex as App

    if ansatz == 'Weighted':
        from classes.ApproximantNN import NN_complex as cl_function
    else:
        from classes.ApproximantNN import classical_complex_Fourier as cl_function

    function_ = function.split('_')
    real = globals()[f"{function_[0]}"]
    imag = globals()[f"{function_[1]}"]

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

    ax_real.scatter(C.domain, np.real(outcomes), color=colors['classical'],
                    label='Classical %s layers' % layers, zorder=layers, marker=marker['classical'])
    ax_imag.scatter(C.domain, np.imag(outcomes), color=colors['classical'],
                    label='Classical %s layers' % layers, zorder=layers, marker=marker['classical'])

    df_q = df_[(df_['layers'] == layers) & (df_['quantum'] == True)]
    k_q = df_q['chi2'].idxmin()
    file_q = 'results/quantum/' + ansatz + '/' + function + '/%s_layers/%s/result.pkl' % (
    layers, df_q.loc[k_q]['trial'])
    x = np.loadtxt(
        'results/quantum/' + ansatz + '/' + function + '/%s_layers/%s/domain.txt' % (layers, df_q.loc[k_q]['trial']))
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

    ax_real.scatter(C.domain, r_outcomes, color=colors['quantum'], label='Quantum %s layers' % layers,
                    zorder=layers, marker=marker['quantum'])
    ax_imag.scatter(C.domain, i_outcomes, color=colors['quantum'], label='Quantum %s layers' % layers,
                    zorder=layers, marker=marker['quantum'])

    file_exp_x = 'results/experiment/' + ansatz + '/' + function + '/%s_layers/results_x.txt'%layers
    domain, x = np.loadtxt(file_exp_x)

    #ax_real.errorbar(domain, 2*x - 1, yerr=error, color=colors['experiment'], zorder=layers, marker=marker['experiment'], ms=5, lw=0, elinewidth=1, capsize=3)
    ax_real.scatter(domain, 2 * x - 1, color=colors['experiment'], zorder=layers, marker=marker['experiment'])
    file_exp_y = 'results/experiment/' + ansatz + '/' + function + '/%s_layers/results_y.txt'%layers
    domain, y = np.loadtxt(file_exp_y)
    '''ax_imag.errorbar(domain, 2 * y - 1, yerr=error, color=colors['experiment'], zorder=layers, marker=marker['experiment'],
                     ms=5, lw=0,
                     elinewidth=1, capsize=3)'''
    ax_imag.scatter(domain, 2 * y - 1, color=colors['experiment'], zorder=layers, marker=marker['experiment'])

    ax_real.plot(C.domain, np.real(C.target), color='black', linewidth=3, label='Target')
    ax_imag.plot(C.domain, np.imag(C.target), color='black', linewidth=3, label='Target')


def paint_real(function, ansatz, ax, df, layers):
    error=0.015
    from classes.ApproximantNN import Approximant_real as App
    if ansatz == 'Weighted':
        from classes.ApproximantNN import NN_real as cl_function
    else:
        from classes.ApproximantNN import classical_real_Fourier as cl_function
    func = globals()[f"{function}"]
    df_ = df[(df['function'] == function) & (df['ansatz'] == ansatz)].copy()

    df_c = df_[(df_['layers'] == layers) & (df_['quantum'] == False)]
    k_c = df_c['chi2'].idxmin()
    file_c = 'results/classical/' + ansatz + '/' + function + '/%s_layers/%s/result.pkl' % (
        layers, df_c.loc[k_c]['trial'])
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

    ax.scatter(C.domain, outcomes, color=colors['classical'], label='Classical',
               zorder=layers, s=25, marker=marker['classical'])


    df_q = df_[(df_['layers'] == layers) & (df_['quantum'] == True)]
    k_q = df_q['chi2'].idxmin()
    file_q = 'results/quantum/' + ansatz + '/' + function + '/%s_layers/%s/result.pkl' % (
    layers, df_q.loc[k_q]['trial'])
    x = np.loadtxt('results/quantum/' + ansatz + '/' + function + '/%s_layers/%s/domain.txt' % (
    layers, df_q.loc[k_q]['trial']))
    with open(file_q, 'rb') as f:
        data_q = pickle.load(f)
    params_q = data_q['x']

    C = App(layers, x, ansatz, func)
    C.set_parameters(params_q)
    outcomes = np.zeros_like(C.domain)
    for j, x in enumerate(C.domain):
        state = C.get_state(x)
        outcomes[j] = C.H.expectation(state)

    ax.scatter(C.domain, outcomes, color=colors['quantum'], label='Quantum',
               zorder=layers, s=25, marker=marker['quantum'])

    file_exp = 'results/experiment/' + ansatz + '/' + function + '/%s_layers/results.txt'%layers
    x, y = np.loadtxt(file_exp)
    y = 2 * y - 1
    C = App(layers, x, ansatz, func)

    if ansatz=='Weighted' and function=='relu':
        ax.errorbar(x, y, yerr=error, color=colors['experiment'], zorder=layers,ms=4, lw=0, elinewidth=1, capsize=5, label='Experiment',
                marker=marker['experiment'])
    else:
        ax.scatter(x, y, color=colors['experiment'], label='Experiment',
                   zorder=layers, s=25, marker=marker['experiment'])

    ax.plot(C.domain, C.target, color='black', linewidth=3, label='Target')


def paint_real_2D(function, ansatz, ax, df, layers, only_target=False):
    df_ = df[(df['function'] == function) & (df['ansatz'] == ansatz)].copy()
    from classes.ApproximantNN import Approximant_real_2D as App
    if ansatz == 'Weighted_2D':
        from classes.ApproximantNN import NN_real_2D as cl_function
    else:
        raise NameError('Ansatz not included in 2D')

    func = globals()[f"{function}"]

    df_c = df_[(df_['layers'] == layers) & (df_['quantum'] == False)]
    k_c = df_c['chi2'].idxmin()
    file_c = 'results/classical/' + ansatz + '/' + function + '/%s_layers/%s/result.pkl' % (
        layers, df_c.loc[k_c]['trial'])
    x = np.loadtxt(
        'results/classical/' + ansatz + '/' + function + '/%s_layers/%s/domain.txt' % (layers, df_c.loc[k_c]['trial']))
    C = App(layers, x, func, ansatz)

    if not only_target:
        with open(file_c, 'rb') as f:
            data_c = pickle.load(f)
        params_c = data_c['x']

        outcomes = cl_function(params_c, layers, x, C.target)

        ax.scatter(C.domain[:, 0], C.domain[:, 1], outcomes, color=colors['classical'],
                   label='Classical %s layers' % layers, zorder=0, s=25)

        df_q = df_[(df_['layers'] == layers) & (df_['quantum'] == True)]
        k_q = df_q['chi2'].idxmin()
        file_q = 'results/quantum/' + ansatz + '/' + function + '/%s_layers/%s/result.pkl' % (
        layers, df_q.loc[k_q]['trial'])

        x = np.loadtxt(
            'results/quantum/' + ansatz + '/' + function + '/%s_layers/%s/domain.txt' % (layers, df_q.loc[k_q]['trial']))
        with open(file_q, 'rb') as f:
            data_q = pickle.load(f)
        params_q = data_q['x']
        C = App(layers, x, func, ansatz)
        C.set_parameters(params_q)
        outcomes = np.zeros(len(C.domain))
        for j, x in enumerate(C.domain):
            state = C.get_state(x)
            outcomes[j] = C.H.expectation(state)

        ax.scatter(C.domain[:, 0], C.domain[:, 1], outcomes, color=colors['quantum'],
                   label='Quantum %s layers' % layers, zorder=2, s=25)

        func = globals()[f"{function}"]

        try:
            file_exp = 'results/experiment/' + ansatz + '/' + function + '/%s_layers/results.txt'%layers
            y = np.loadtxt(file_exp)
            y = y.transpose()
            y = 2 * y.flatten() - 1

            ax.scatter(C.domain[:, 0], C.domain[:, 1], y, color=colors['experiment'], label='Experiment %s layers' % layers, zorder=layers, s=25)
        except:
            pass

    cmap = plt.get_cmap('Greys')
    ax.plot_trisurf(C.domain[:, 0], C.domain[:, 1], C.target-0.1, cmap=cmap, vmin=-4, vmax=1, alpha=0.75)


cmap = {'target':plt.get_cmap('Greys'), 'classical':plt.get_cmap('Blues'), 'experiment':plt.get_cmap('Greens'), 'quantum':plt.get_cmap('Reds')}
#cmap = {'target':plt.get_cmap('viridis'), 'classical':plt.get_cmap('inferno'), 'experiment':plt.get_cmap('PiYG'), 'quantum':plt.get_cmap('bwr')}
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def paint_real_2D_2(function, ansatz, ax, df, layers, data):
    from classes.ApproximantNN import Approximant_real_2D as App
    if ansatz == 'Weighted_2D':
        from classes.ApproximantNN import NN_real_2D as cl_function
    else:
        raise NameError('Ansatz not included in 2D')

    func = globals()[f"{function}"]
    df_ = df[(df['function'] == function) & (df['ansatz'] == ansatz)].copy()
    if data == 'target':
        x = np.loadtxt(
            'results/classical/' + ansatz + '/' + function + '/%s_layers/%s/domain.txt' % (
            layers, 0))
        C = App(layers, x, func, ansatz)
        X, Y = np.meshgrid(np.linspace(-5, 5, 25), np.linspace(-5, 5, 25))
        outcomes = C.target
        size = len(outcomes)
        outcomes = outcomes.reshape((int(np.sqrt(size)), int(np.sqrt(size))))


    if data == 'classical':
        df_c = df_[(df_['layers'] == layers) & (df_['quantum'] == False)]
        k_c = df_c['chi2'].idxmin()
        file_c = 'results/classical/' + ansatz + '/' + function + '/%s_layers/%s/result.pkl' % (
            layers, df_c.loc[k_c]['trial'])
        x = np.loadtxt(
            'results/classical/' + ansatz + '/' + function + '/%s_layers/%s/domain.txt' % (layers, df_c.loc[k_c]['trial']))
        C = App(layers, x, func, ansatz)


        with open(file_c, 'rb') as f:
            data_c = pickle.load(f)
        params_c = data_c['x']

        outcomes = cl_function(params_c, layers, x, C.target)
        X, Y = np.meshgrid(np.linspace(-5, 5, 25), np.linspace(-5, 5, 25))
        size = len(outcomes)
        outcomes = outcomes.reshape((int(np.sqrt(size)), int(np.sqrt(size))))

    if data == 'quantum':
        df_q = df_[(df_['layers'] == layers) & (df_['quantum'] == True)]
        k_q = df_q['chi2'].idxmin()
        file_q = 'results/quantum/' + ansatz + '/' + function + '/%s_layers/%s/result.pkl' % (
            layers, df_q.loc[k_q]['trial'])

        x = np.loadtxt(
            'results/quantum/' + ansatz + '/' + function + '/%s_layers/%s/domain.txt' % (
            layers, df_q.loc[k_q]['trial']))
        with open(file_q, 'rb') as f:
            data_q = pickle.load(f)
        params_q = data_q['x']
        C = App(layers, x, func, ansatz)
        C.set_parameters(params_q)
        outcomes = np.zeros(len(C.domain))
        for j, x in enumerate(C.domain):
            state = C.get_state(x)
            outcomes[j] = C.H.expectation(state)

        X, Y = np.meshgrid(np.linspace(-5, 5, 25), np.linspace(-5, 5, 25))
        size = len(outcomes)
        outcomes = outcomes.reshape((int(np.sqrt(size)), int(np.sqrt(size))))

    if data == 'experiment':
        func = globals()[f"{function}"]

        file_exp = 'results/experiment/' + ansatz + '/' + function + '/%s_layers/results.txt' % layers
        y = np.loadtxt(file_exp)
        y = y.transpose()
        outcomes = 2 * y.flatten() - 1

        X, Y = np.meshgrid(np.linspace(-5, 5, 25), np.linspace(-5, 5, 25))
        size = len(outcomes)
        outcomes = outcomes.reshape((int(np.sqrt(size)), int(np.sqrt(size))))

    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=-1., vmax=1.)
    cf = ax.contourf(X, Y, outcomes, cmap=cmap[data], norm=norm, levels = np.linspace(-1,1,21))
    ax.contour(cf, colors='k', linewidths=1)
    return cf


#plt.style.use('seaborn')
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(24,12), sharex=True, sharey=True)


i = 0
for ansatz in ['Fourier', 'Weighted']:
    for function in ['tanh', 'step', 'poly', 'relu']:
        ax = axs.flatten()[i]
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=18)

        paint_real(function, ansatz, ax, df, L)
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
handles.append(mlines.Line2D([], [], color=colors['classical'], markersize=10, label= 'Classical', linewidth=0, marker=marker['classical']))
handles.append(mlines.Line2D([], [], color=colors['quantum'], markersize=10, label= 'Simulation', linewidth=0, marker=marker['quantum']))
handles.append(mlines.Line2D([], [], color=colors['experiment'], markersize=10, label='Experiment',  linewidth=0, marker=marker['experiment']))
handles.append(mlines.Line2D([], [], color='black', markersize=0, label='Target', linewidth=4))
fig.legend(handles = handles, bbox_to_anchor=(0.86, 0.6, 0.12, .4), loc='lower left', borderaxespad=0., mode='expand', fontsize=20)

fig.savefig('functions_reals_%sL.pdf'%L)



fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,12), sharex=True, sharey=True)


i = 0
function = 'tanh_relu'
for ansatz in ['Fourier','Weighted']:
    ax = axs[i]
    ax[0].grid(True)
    ax[0].tick_params(axis='both', which='major', labelsize=18)
    ax[1].grid(True)
    ax[1].tick_params(axis='both', which='major', labelsize=18)
    paint_complex(function, ansatz, ax[0], ax[1], df, L)
    i += 1

    pos = ax[0].get_position()
    pos.x1 -= 0.02
    ax[0].set_position(pos)
    pos = ax[1].get_position()
    pos.x0 -= 0.05
    pos.x1 -= 0.07
    ax[1].set_position(pos)


axs.flatten()[0].set_ylabel(r'$f(x)$', fontsize=24)
axs.flatten()[2].set_ylabel(r'$f(x)$', fontsize=24)

axs.flatten()[2].set_xlabel(r'$x$', fontsize=24)
axs.flatten()[3].set_xlabel(r'$x$', fontsize=24)

fig.text(0.005, 0.65, 'Fourier', rotation='vertical', fontsize=30, fontweight='bold')
fig.text(0.005, 0.26, 'UAT', rotation='vertical', fontsize=30, fontweight='bold')

fig.text(0.15, 0.9, r'Real part = $\tanh(5x)$', fontsize=30)
fig.text(0.52, 0.9, r'Imag part = ${\rm ReLU}(x)$', fontsize=30)

fig.legend(handles = handles, bbox_to_anchor=(0.835, 0.48, 0.155, .4), loc='lower left', borderaxespad=0., mode='expand', fontsize=18)

fig.savefig(function + '_complex_%sL.pdf'%L)


fig = plt.figure(figsize=(15,14))
i = 1
ansatz = 'Weighted_2D'
for function in ['Himmelblau','Brent','Adjiman','Threehump']:
    ax = fig.add_subplot(2, 2, i, projection='3d')
    paint_real_2D(function.lower(), ansatz, ax, df, L, only_target=True)
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

handles.append(mlines.Line2D([], [], color=colors['classical'], markersize=10, label='Classical' , linewidth=0, marker=marker['classical']))
handles.append(mlines.Line2D([], [], color=colors['quantum'], markersize=10, label='Quantum' ,
                             linewidth=0, marker=marker['quantum']))
handles.append(mlines.Line2D([], [], color=colors['experiment'], markersize=10, label='Experiment' ,
                             linewidth=0, marker=marker['experiment']))
fig.legend(handles = handles, bbox_to_anchor=(0.4, -0.07, 0.18, .2),borderaxespad=0., mode='expand', fontsize=20, ncol=1)

#fig.text(0.35, 0.04, '%s layers'%L, fontsize=20, fontweight='bold')
fig.savefig('model_2D_2x2.pdf')


fig = plt.figure(figsize=(15,14))
i = 1
ansatz = 'Weighted_2D'
function = 'Himmelblau'
data = ['target', 'classical', 'quantum', 'experiment']
titles = {'target':'Target', 'classical':'Classical', 'quantum':'Simulation', 'experiment':'Experiment'}
for d in data:
    ax = fig.add_subplot(2, 2, i)
    cf = paint_real_2D_2(function.lower(), ansatz, ax, df, L, d)
    if i in [3,4]:
        pos = ax.get_position()
        pos.y0 -= .065
        pos.y1 += 0.005
        ax.set_position(pos)
        ax.set_xlabel('x', fontsize=20)
        ax.set_xticks([-5,0,5])
    else:
        ax.set_xticks([])
        pos = ax.get_position()
        pos.y0 -= 0.02
        pos.y1 += 0.05
        ax.set_position(pos)
    if i in [1,3]:
        ax.set_ylabel('y', fontsize=20)
        ax.set_yticks([-5, 0, 5])
        pos = ax.get_position()
        pos.x0 -= 0.07
        pos.x1 -= 0.01
        ax.set_position(pos)
    else:
        ax.set_yticks([])
        pos = ax.get_position()
        pos.x1 += 0.04
        pos.x0 -= 0.02
        ax.set_position(pos)
    ax.set_title(titles[d], fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)

    axins = inset_axes(ax,
                       width="3%",  # width = 5% of parent_bbox width
                       height="90%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.01, 0.05, 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0,
                       )

    # Controlling the placement of the inset axes is basically same as that
    # of the legend.  you may want to play with the borderpad value and
    # the bbox_to_anchor coordinate.

    cbar = fig.colorbar(cf, cax=axins, ticks=[-1., -.5, 0, .5, 1.])
    cbar.ax.tick_params(labelsize=15)
    #fig.colorbar(cf, ax=ax)
    i += 1

fig.suptitle('Himmelblau(x, y)', fontsize=36)
fig.savefig(function + '_2D_%sL_2.pdf'%L)


fig = plt.figure(figsize=(20,5.6))
i = 1
ansatz = 'Weighted_2D'
function = 'Himmelblau'
data = ['target', 'classical', 'quantum', 'experiment']
titles = {'target':'Target', 'classical':'Classical', 'quantum':'Simulation', 'experiment':'Experiment'}
for d in data:
    ax = fig.add_subplot(1, 4, i)
    cf = paint_real_2D_2(function.lower(), ansatz, ax, df, L, d)
    if i == 1:
        ax.set_ylabel('y', fontsize=20)
        ax.set_yticks([-5, 0, 5])
    else:
        ax.set_yticks([])
    ax.set_xlabel('x', fontsize=20)
    ax.set_xticks([-5, 0, 5])

    pos = ax.get_position()
    pos.x0 = 0.245 * (i-1) + 0.04
    pos.x1 = pos.x0 + 0.185
    pos.y1 = pos.y0 + 0.7
    ax.set_position(pos)


    ax.set_title(titles[d], fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)

    axins = inset_axes(ax,
                       width="3%",  # width = 5% of parent_bbox width
                       height="90%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.01, 0.05, 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0,
                       )

    # Controlling the placement of the inset axes is basically same as that
    # of the legend.  you may want to play with the borderpad value and
    # the bbox_to_anchor coordinate.

    cbar = fig.colorbar(cf, cax=axins, ticks=[-1., -.5, 0, .5, 1.])
    cbar.ax.tick_params(labelsize=15)
    #fig.colorbar(cf, ax=ax)
    i += 1

fig.suptitle('Himmelblau(x, y)', fontsize=36)
fig.savefig(function + '_2D_%sL_3.pdf'%L)