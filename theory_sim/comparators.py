from classes import *
import matplotlib.pyplot as plt


def increase_layers(algorithm, max_layers, domain, f, phase=None):
    init_point = np.random.rand(3)
    for l in range(1, max_layers + 1):
        try:
            alg = algorithm(l, domain, f)
        except:
            alg = algorithm(l, domain, f, phase)
        alg.update_parameters(init_point_q)
        res_q = alg.find_optimal_parameters(gens=100, tol=1e-8, verbose=0)  # Hay que encontrar un método para poner las generaciones de manera apropiada, ni demasiadas ni demasiado pocas
        # PROBLEMÓN: cma tal y como está es una caja negra
        # idea: copiar el archivo de cma a un archivo auxiliar y manipular el código
        print(res_q)
        init_point = np.concatenate((np.zeros((1, 3)), alg.params), axis=0)


def comparator_NN(max_layers, domain, f): # Quitar la parte gráfica y guardar solo los jso
    init_point_q = np.random.rand(3)
    init_point_ = np.random.rand(3)
    for l in range(1, max_layers + 1):
        q_nn = Approximant_NN(l, domain, f)
        q_nn.update_parameters(init_point_q)
        res_q = q_nn.find_optimal_parameters(gens=1000)  # Hay que encontrar un método para poner las generaciones de manera apropiada, ni demasiadas ni demasiado pocas
                                                                # PROBLEMÓN: cma tal y como está es una caja negra
                                                                # idea: copiar el archivo de cma a un archivo auxiliar y manipular el código
        print(res_q)
        init_point_q = np.concatenate((q_nn.params, np.zeros((1,3))), axis=0)  # The positions is not relevant

        nn = NN(l, domain, f)
        nn.update_parameters(init_point_)
        res = nn.find_optimal_parameters(gens=100)
        init_point_ = np.concatenate((nn.params, np.zeros((1,3))), axis=0)  # The positions is not relevant
        print(res)


    fig, ax = plt.subplots()
    ax.scatter(np.arange(1, max_layers + 1), f_q, label='Quantum')
    ax.scatter(np.arange(1, max_layers + 1), f_, label='Neural Network')
    ax.legend() # Poner las figuras bonitas tiene que ir con el código que sigue en Barcelona, no es necesario perder el tiempo en Abu Dhabi
    fig.savefig('inc_layers/nn_' + f.name + '_errors.png')


def comparator_Fourier(max_layers, domain, f, painting_layers = []):
    if len(painting_layers) > 0:
        fig, ax = plt.subplots()
        ax.plot(domain, f(domain))

    f_q = np.zeros(max_layers)
    f_ = np.zeros(max_layers)
    for l in range(1, max_layers + 1):
        q_nn = Approximant_Fourier(l, domain, f)
        res_q = q_nn.find_optimal_parameters()
        f_q[l-1] = res_q.fun
        #q_nn.run()

        F = Fourier(l, domain, f)
        F.find_optimal_parameters()
        F.run()
        f_[l-1] = F.approximation()

        if l in painting_layers:
            ax.plot(domain, np.abs(q_nn.final_states[:, 1])**2, linestyle='--', label='{} layers: quantum'.format(l))
            ax.plot(domain, F.final_states, linestyle=':', label='{} layers: NN'.format(l))
            ax.legend()

    try:
        fig.savefig('inc_layers/F_' + f.name + '_function.png')
    except:
        pass

    fig, ax = plt.subplots()
    ax.scatter(np.arange(1, max_layers + 1), f_q, label='Quantum')
    ax.scatter(np.arange(1, max_layers + 1), f_, label='Fourier')
    ax.legend()
    fig.savefig('inc_layers/F_' + f.name + '_errors.png')

def comparator_NN_complex(max_layers, domain, f, phase, painting_layers=[]):
    if len(painting_layers) > 0:
        fig, axs = plt.subplots(nrows=2)
        axs[0].plot(domain, f(domain))
        axs[1].plot(domain, phase(2 * np.pi * domain))

    f_q = np.zeros(max_layers)
    f_ = np.zeros(max_layers)
    for l in range(1, max_layers + 1):
        q_nn = Approximant_NN_complex(l, domain, f, phase)
        res_q = q_nn.find_optimal_parameters()
        f_q[l - 1] = res_q.fun
        q_nn.run()

        nn = NN_complex(l, domain, f, phase)
        res = nn.find_optimal_parameters()
        #nn.run()
        f_[l - 1] = res.fun

        if l in painting_layers:
            axs[0].plot(domain, np.abs(q_nn.final_states[:, 1]) ** 2, linestyle='--',
                    label='{} layers: quantum'.format(l))
            axs[0].plot(domain, np.abs(nn.final_states), linestyle=':', label='{} layers: NN'.format(l))
            axs[0].legend()

            angles = np.angle(q_nn.final_states[:, 1] ** 2)
            angles[angles < 0] = 2 * np.pi + angles[angles < 0]
            axs[1].plot(domain, angles, linestyle='--',
                        label='{} layers: quantum'.format(l))

            angles = np.angle(nn.final_states)
            angles[angles < 0] = 2 * np.pi + angles[angles < 0]
            axs[1].plot(domain, angles, linestyle=':', label='{} layers: NN'.format(l))
            axs[1].legend()

    try:
        fig.savefig('inc_layers/nn_complex_' + f.name + '_' + phase.name + '_function.png')
    except:
        pass

    fig, ax = plt.subplots()
    ax.scatter(np.arange(1, max_layers + 1), f_q, label='Quantum')
    ax.scatter(np.arange(1, max_layers + 1), f_, label='Neural Network')
    ax.legend()
    fig.savefig('inc_layers/nn_complex_' + f.name + '_' + phase.name + '_errors.png')

def comparator_Fourier_complex(max_layers, domain, f, phase, painting_layers=[]):
    if len(painting_layers) > 0:
        fig, axs = plt.subplots(nrows=2)
        axs[0].plot(domain, f(domain))
        axs[1].plot(domain, phase(2 * np.pi * domain))

    f_q = np.zeros(max_layers)
    f_ = np.zeros(max_layers)
    for l in range(1, max_layers + 1):
        q_nn = Approximant_Fourier_complex(l, domain, f, phase)
        res_q = q_nn.find_optimal_parameters()
        f_q[l - 1] = res_q.fun
        q_nn.run()

        F = Fourier_complex(l, domain, f, phase)
        res = F.find_optimal_parameters()
        F.run()
        f_[l - 1] = F.approximation()
        if l in painting_layers:
            axs[0].plot(domain, np.abs(q_nn.final_states[:, 1]) ** 2, linestyle='--',
                    label='{} layers: quantum'.format(l))
            axs[0].plot(domain, np.abs(F.final_states), linestyle=':', label='{} layers: NN'.format(l))
            axs[0].legend()

            angles = np.angle(q_nn.final_states[:, 1] ** 2)
            angles[angles < 0] = 2 * np.pi + angles[angles < 0]
            axs[1].plot(domain, angles, linestyle='--',
                        label='{} layers: quantum'.format(l))

            angles = np.angle(F.final_states)
            angles[angles < 0] = 2 * np.pi + angles[angles < 0]
            axs[1].plot(domain, angles, linestyle=':', label='{} layers: NN'.format(l))
            axs[1].legend()

    try:
        fig.savefig('inc_layers/F_complex_' + f.name + '_' + phase.name + '_function.png')
    except:
        pass

    fig, ax = plt.subplots()
    ax.scatter(np.arange(1, max_layers + 1), f_q, label='Quantum')
    ax.scatter(np.arange(1, max_layers + 1), f_, label='Fourier')
    ax.legend()
    fig.savefig('inc_layers/F_complex_' + f.name + '_' + phase.name + '_errors.png')





