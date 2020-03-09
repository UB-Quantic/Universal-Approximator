from classes import *
import matplotlib.pyplot as plt

def calculator_NN(layers, domain, f, noisy=False, samples=10000):
    fig, ax = plt.subplots()
    ax.plot(domain, f(domain))
    q_nn = Approximant_NN(layers, domain, f)
    res_q = q_nn.find_optimal_parameters(noisy=noisy, samples=samples)

    nn = NN(layers, domain, f)
    res = nn.find_optimal_parameters()
    if not noisy:
        ax.plot(domain, np.abs(q_nn.final_states[:, 1]) ** 2, linestyle='--', label='Quantum')
    else:
        s = q_nn.sampling(samples)
        ax.plot(domain, s, linestyle='--', label='Quantum')
    ax.plot(domain, nn.final_states, linestyle=':', label='NN')
    ax.legend()

    if not noisy:
        fig.savefig('single_case/nn_' + f.name + '_%s'%(layers,) + '.png')
    else:
        fig.savefig('single_case/nn_' + f.name + '_%s_%s_noisy.png' % (layers, samples))

    plt.close()
    return res_q.fun, res.fun


def calculator_Fourier(layers, domain, f, noisy=False, samples=10000):
    fig, ax = plt.subplots()
    ax.plot(domain, f(domain))
    q_nn = Approximant_NN(layers, domain, f)
    res_q = q_nn.find_optimal_parameters(noisy=noisy, samples=samples)

    F = Fourier(layers, domain, f)
    F.find_optimal_parameters()
    F.run()
    res = F.approximation()
    if not noisy:
        ax.plot(domain, np.abs(q_nn.final_states[:, 1]) ** 2, linestyle='--', label='Quantum')
    else:
        s = q_nn.sampling(samples)
        ax.plot(domain, s, linestyle='--', label='Quantum')
    ax.plot(domain, F.final_states, linestyle=':', label='NN')
    ax.legend()

    if not noisy:
        fig.savefig('single_case/F_' + f.name + '_%s'%(layers,) + '.png')
    else:
        fig.savefig('single_case/F_' + f.name + '_%s_%s_noisy.png' % (layers, samples))

    plt.close()
    return res_q.fun, res


def calculator_NN_complex(layers, domain, f, phase):
    fig, axs = plt.subplots(nrows=2)
    axs[0].plot(domain, f(domain))
    axs[1].plot(domain, phase(2 * np.pi * domain))
    q_nn = Approximant_NN_complex(layers, domain, f, phase)
    res_q = q_nn.find_optimal_parameters()

    nn = NN_complex(layers, domain, f, phase)
    res = nn.find_optimal_parameters()

    axs[0].plot(domain, np.abs(q_nn.final_states[:, 1]) ** 2, linestyle='--',
                label='{} layers: quantum'.format(layers))
    axs[0].plot(domain, np.abs(nn.final_states), linestyle=':', label='{} layers: NN'.format(layers))
    axs[0].legend()

    angles = np.angle(q_nn.final_states[:, 1] ** 2)
    angles[angles < 0] = 2 * np.pi + angles[angles < 0]
    axs[1].plot(domain, angles, linestyle='--',
                label='{} layers: quantum'.format(layers))

    angles = np.angle(nn.final_states)
    angles[angles < 0] = 2 * np.pi + angles[angles < 0]
    axs[1].plot(domain, angles, linestyle=':', label='{} layers: NN'.format(layers))
    axs[1].legend()

    fig.savefig('single_case/nn_complex_' + f.name + '_' + phase.name + '_%s.png' %layers)

    print('Cost function Quantum = %s' %res_q.fun)
    print('Cost function NN = %s' % res.fun)
