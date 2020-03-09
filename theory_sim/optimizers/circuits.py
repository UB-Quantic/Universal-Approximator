from qcgpu import State
import numpy as np

def create_theta(qubits, layers):
    theta = np.empty(layers*2 + 1, qubits, 3)

    return theta


def circuit(theta):
    l, qubits, n = theta.shape
    layers = (l - 1) // 2
    qc = State(qubits)
    for l in range(layers):
        for q in range(qubits):
            qc.u(q, theta[2*l, q, 0], theta[2*l, q, 1], theta[2*l, q, 1])
        for q in range(0, qubits, 2):
            qc.cz(q, q+1)
        for q in range(qubits):
            qc.u(q, theta[2 * l + 1, q, 0], theta[2 * l + 1, q, 1], theta[2 * l + 1, q, 1])
        for q in range(0, qubits, 2):
            qc.cz(q + 1, (q + 2)%qubits)

    for q in range(qubits):
        qc.u(q, theta[-1, q, 0], theta[-1, q, 1], theta[-1, q, 1])

    return qc


def cost_KL(p, q, epsilon=1e-6):
    c = np.sum([p[i] * np.log(p[i] / max(q[i], epsilon)) for i in range(len(p))])

    return c / len(p)

def cost_square(p, q):
    c = 1 / 2 * np.sum((p - q)**2)

    return c / len(p)

def gaussian_distr(qubits, n_sigmas=3):
    x = np.linspace(0, n_sigmas, 2**(qubits-1))
    probs_ = 1 / np.sqrt(2 * np.pi) * np.exp( - 0.5 * np.power(x, 2))
    probs__ = np.sort(probs_)

    prob = np.concatenate((probs__, probs_))
    return prob / np.sum(prob)


def get_probabilities_from_measurement(theta, samples):
    qc = circuit(theta)
    meas = qc.measure(samples)
    probs = np.zeros_like(qc.probabilities())
    for k in list(meas.keys()):
        probs[int(k, 2)] = meas[k]

    return probs / np.sum(probs)

def cost(theta, samples, original_distribution, function='square'):
    if function=='square':
        cost_function = cost_square
    if function=='KL':
        cost_function = cost_KL
    q = get_probabilities_from_measurement(theta, samples)
    return cost_function(original_distribution, q)
