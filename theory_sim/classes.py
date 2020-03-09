import numpy as np
from scipy.integrate import trapz
from opt_algorithms import adam_optimizer_noisy, adam_optimizer, _evol, _cma, adam_spsa_optimizer
import json
from scipy.optimize import minimize, basinhopping, differential_evolution
import os

np.random.seed(0)  #Seed 0, crucial para repetir experimentos

def step(x):
    step.name = 'step'
    return 0.5 * (np.sign(x) + 1)

def cosine(x):
    cosine.name = 'cosine'
    return 0.5 * (np.cos(2*np.pi*x) + 1)

def sigmoid(x):
    sigmoid.name = 'sigmoid'

    return 1 / (1 + np.exp(-x))

def tanh(x):
    tanh.name = 'tanh'

    return 0.5 * (np.tanh(x) + 1)

def relu(x):
    relu.name = 'relu'

    return np.clip(x, 0, np.max(x))

def poly(x):
    poly.name= 'poly'

    return np.abs(3*x**3 * (1 - x**4))

def M3(x, params_3):
    params_3 = 0.5 * params_3
    m = np.array([[np.exp(1j * params_3[2]) * np.cos(params_3[0] * x + params_3[1]), -np.exp(1j * params_3[2]) * np.sin(params_3[0] * x + params_3[1])],
                  [np.exp(-1j * params_3[2]) * np.sin(params_3[0] * x + params_3[1]), np.exp(-1j * params_3[2]) * np.cos(params_3[0] * x + params_3[1])]],
                 dtype='complex')
    return m

def M4(x, params_4):
    params_4 = 0.5 * params_4
    m = np.array([[np.exp(1j * (params_4[2] * x + params_4[3])) * np.cos(params_4[0] * x + params_4[1]),
                   -np.exp(1j * (params_4[2] * x + params_4[3])) * np.sin(params_4[0] * x + params_4[1])],
                  [np.exp(-1j * (params_4[2] * x + params_4[3])) * np.sin(params_4[0] * x + params_4[1]),
                   np.exp(-1j * (params_4[2] * x + params_4[3])) * np.cos(params_4[0] * x + params_4[1])]],
                 dtype='complex')
    return m

def M5(x, params_5):
    params_5 = 0.5 * params_5
    m = np.array([[np.cos(params_5[4]) * np.cos(params_5[3]) * np.exp(1j * (params_5[1] + params_5[0] * x))
                   - np.sin(params_5[4]) * np.sin(params_5[3]) * np.exp(1j * (params_5[2] - params_5[0] * x)),
                   -np.cos(params_5[4]) * np.sin(params_5[3]) * np.exp(1j * (params_5[1] + params_5[0] * x))
                   - np.sin(params_5[4]) * np.cos(params_5[3]) * np.exp(1j * (params_5[2] - params_5[0] * x))],
                  [np.sin(params_5[4]) * np.cos(params_5[3]) * np.exp(1j * (-params_5[2] + params_5[0] * x))
                   + np.cos(params_5[4]) * np.sin(params_5[3]) * np.exp(1j * (-params_5[1] - params_5[0] * x)),
                   - np.sin(params_5[4]) * np.sin(params_5[3]) * np.exp(1j * (-params_5[2] + params_5[0] * x))
                   + np.cos(params_5[4]) * np.cos(params_5[3]) * np.exp(1j * (-params_5[1] - params_5[0] * x))]],
                 dtype='complex')

    return m
def state(x, theta):
    params=theta.shape[-1]
    if params == 3:
        M = M3(x, theta[0, :])
        for i in range(1, theta.shape[0]):
            M = M3(x, theta[i, :]) @ M

    if params == 4:
        M = M4(x, theta[0, :])
        for i in range(1, theta.shape[0]):
            M = M4(x, theta[i, :]) @ M

    if params == 5:
        M = M5(x, theta[0, :])
        for i in range(1, theta.shape[0]):
            M = M3(x, theta[i, :]) @ M
    s = np.array([[1],[0]])
    s = M @ s

    return s

class Approximant_NN:
    def __init__(self, layers, domain, f):
        self.params = np.random.randn(layers, 3)
        self.domain = domain
        self.function = f(self.domain)
        self.final_states = np.zeros((len(self.domain), 2), dtype=complex)
        self.layers = layers
        self.f = f
        self.name = 'q_NN'


    def update_parameters(self, new_parameters):
        self.params = new_parameters

    def run(self, noisy=False, samples=10000, batch=1):
        if batch != 1:
            num_x = int(batch * len(self.domain))
            X = np.random.choice(self.domain, num_x, replace=False)
        else:
            X = self.domain
        if not noisy:
            for i, x in enumerate(X):
                self.final_states[i] = state(x, self.params).flatten()
        else:
            self.run()
            probs = np.abs(self.final_states[:, 1]) ** 2
            sampling = np.random.binomial(n=1, p=probs, size=(samples, len(probs)))
            sampling = np.sum(sampling, axis=0) / samples
            self.outcomes = sampling


    def find_optimal_parameters(self, init_point=None, noisy=False, samples=10000, batch=1, gens=100, tol=1e-8, verbose=True):
        if init_point is None:
            init_point = self.params.flatten()

        if not noisy:
            # result = _evol('nn', self._minim_function, self.layers, gens, N=100, tol=tol, verbose=verbose)
            # result = _cma('nn', self._minim_function, init_point, self.layers, gens, tol=tol, verbose=verbose)
            # result = minimize(self._minim_function, init_point, method='bfgs', options={"disp":verbose})
            # result = adam_optimizer(self._minim_function, init_point, gens=gens)
            result = adam_spsa_optimizer(self._minim_function, init_point, batch, tol=tol)
            # result = basinhopping(self._minim_function, init_point, disp=verbose, minimizer_kwargs={'method':'cobyla'}, niter_success=3)
            # result = differential_evolution(self._minim_function, [(-np.pi, np.pi)] * len(init_point), disp=verbose)
            print(result)
            result['x'] = list(result['x'])
            try:
                result['jac'] = list(result['jac'])
                del result['message']
                del result['hess_inv']
            except: pass

            folder = fold_name(self.name, self.f)
            filename = folder + '/%s_exact.txt' % self.layers
            #save_dict(result, folder, filename)
            return result


        else:
            # result = _cma_nn(self._minim_function_noisy, init_point, samples, self.layers, gens, tol=tol, verbose=verbose)
            # result = minimize(self._minim_function_noisy, init_point, method='powell', options={'disp':verbose, 'ftol':1 / np.sqrt(samples) / len(self.domain)})
            result = adam_optimizer_noisy(self._minim_function_noisy, init_point, samples)
            # result = basinhopping(self._minim_function_noisy, init_point, niter_success=3, minimizer_kwargs={'args':(samples), 'method':'powell', 'options':{'ftol':1 / np.sqrt(samples) / len(self.domain)}}, disp=verbose)
            print(result)
            result['x'] = list(result['x'])
            #del result['message']
            #del result['direc']

        return result

    def _minim_function(self, params, batch=1):
        params = np.asarray(params)
        params = params.reshape((self.layers, 3))
        self.update_parameters(params)
        self.run()
        chi = np.mean((np.abs(self.final_states[:, 1])**2 - self.function) ** 2)
        return chi

    def _minim_function_noisy(self, params, samples=10000):
        params = np.asarray(params)
        params = params.reshape((self.layers, 3))
        self.update_parameters(params)
        self.run()
        probs = np.abs(self.final_states[:, 1])**2
        sampling = np.random.binomial(n=1, p=probs, size=(samples, len(probs)))
        sampling = np.sum(sampling, axis=0) / samples
        chi = np.mean((sampling - self.function) ** 2)

        return chi

    def sampling(self, samples=10000):
        self.run()
        probs = np.abs(self.final_states[:, 1]) ** 2
        sampling = np.random.binomial(n=1, p=probs, size=(samples, len(probs)))
        sampling = np.sum(sampling, axis=0) / samples
        return sampling

class Approximant_NN_complex:
    def __init__(self, layers, domain, f, phase):
        self.params = np.random.randn(layers, 3)
        self.domain = domain
        self.function = f(self.domain) * np.exp(1j * 2 * np.pi * phase(self.domain))
        self.final_states = np.zeros((len(self.domain), 2), dtype=complex)
        self.layers = layers
        self.name = 'q_NN_complex'
        self.f = f
        self.phase = phase

    def update_parameters(self, new_parameters):
        self.params = new_parameters

    def run(self):
        for i, x in enumerate(self.domain):
            self.final_states[i] = state(x, self.params).flatten()

        for i, f in enumerate(self.final_states):
            self.final_states[i] = f * np.exp(-1j * np.angle(f[0]))

    def find_optimal_parameters(self, init_point=None, gens=100, tol=1e-8, verbose=1):
        if init_point is None:
            init_point = self.params.flatten()

        # result = minimize(self._minim_function, params_aux, method='L-BFGS-B')
        result = _cma_nn(self._minim_function, init_point, self.layers, gens, tol=tol, verbose=verbose)
        filename = 'results/' + self.name + '/' + self.f.name + '/%s_exact.txt' % self.layers
        with open(filename, 'w') as outfile:
            json.dump(result, outfile)

        return result

    def _minim_function(self, params):
        params = np.asarray(params)
        params = params.reshape((self.layers, 3))
        self.update_parameters(params)
        self.run()
        chi = np.mean(np.abs(self.final_states[:, 1]**2 - self.function) ** 2)
        return (chi,)

class Approximant_Fourier:
    def __init__(self, layers, domain, f):
        self.params = np.random.randn(layers, 5)
        self.domain = domain
        self.function = f(self.domain)
        self.final_states = np.zeros((len(self.domain), 2), dtype=complex)
        self.layers = layers
        self.name = 'q_Fourier'
        self.f = f

    def update_parameters(self, new_parameters):
        self.params = new_parameters

    def run(self):
        for i, x in enumerate(self.domain):
            self.final_states[i] = state(x, self.params).flatten()

    def find_optimal_parameters(self, init_point=None, gens=100, tol=1e-8, verbose=1):
        if init_point is None:
            init_point = self.params.flatten()

        result = minimize(self._minim_function, init_point, method='BFGS')
        result['x'] = list(result['x'])
        try:
            result['jac'] = list(result['jac'])
            del result['message']
            del result['hess_inv']
        except:
            pass
        # result = _cma_fourier(self._minim_function, init_point, self.layers, gens, tol=tol, verbose=verbose)
        folder = fold_name(self.name, self.f)
        filename = folder + '/%s_exact.txt' % self.layers
        save_dict(result, folder, filename)

        print(result['fun'])
        return result

    def _minim_function(self, params):
        params = np.asarray(params)
        params = params.reshape((self.layers, 5))
        self.update_parameters(params)
        self.run()
        chi = np.mean((np.abs(self.final_states[:, 1])**2 - self.function) ** 2)
        return chi

class Approximant_Fourier_complex:
    def __init__(self, layers, domain, f, phase):
        self.params = np.random.randn(layers, 5)
        self.domain = domain
        self.function = f(self.domain) * np.exp(1j * 2 * np.pi * phase(self.domain))
        self.final_states = np.zeros((len(self.domain), 2), dtype=complex)
        self.layers = layers
        self.name = 'q_Fourier_complex'
        self.f = f
        self.phase = phase

    def reset(self):
        self.set_state(np.array([1., 0.]))

    def update_parameters(self, new_parameters):
        self.params = new_parameters

    def run(self):
        for i, x in enumerate(self.domain):
            self.final_states[i] = state(x, self.params).flatten()

        for i, f in enumerate(self.final_states):
            self.final_states[i] = f * np.exp(-1j * np.angle(f[0]))


    def find_optimal_parameters(self, init_point=None, gens=100, tol=1e-8, verbose=1):
        if init_point is None:
            init_point = self.params.flatten()

        # result = minimize(self._minim_function, params_aux, method='BFGS')
        result = _cma_fourier(self._minim_function, init_point, self.layers, gens, tol=tol, verbose=verbose)
        filename = 'results/' + self.name + '/' + self.f.name + '/%s_exact.txt' % self.layers
        with open(filename, 'w') as outfile:
            json.dump(result, outfile)

        return result

    def _minim_function(self, params):
        params = np.asarray(params)
        params = params.reshape((self.layers, 5))
        self.update_parameters(params)
        self.run()
        chi = np.mean(np.abs(self.final_states[:, 1]**2 - self.function) ** 2)
        return (chi,)


class NN:
    def __init__(self, layers, domain, f):
        self.params = np.random.randn(layers, 3)
        self.domain = domain
        self.function = f(self.domain)
        self.final_states = np.zeros_like(self.domain)
        self.layers = layers
        self.name = 'NN'
        self.f = f

    def update_parameters(self, new_parameters):
        self.params = new_parameters

    def run(self):
        self.final_states = np.zeros_like(self.domain)
        for i, x in enumerate(self.domain):
            for l in range(self.layers):
                self.final_states[i] += self.params[l, 2] * np.cos(self.params[l, 0] * x + self.params[l, 1])

    def find_optimal_parameters(self, init_point=None, gens=100, tol=1e-8, verbose=0):
        if init_point is None:
            init_point = self.params.flatten()

        result = minimize(self._minim_function, init_point, method='BFGS')
        result['x'] = list(result['x'])
        try:
            result['jac'] = list(result['jac'])
            del result['message']
            del result['hess_inv']
        except:
            pass
        # result = _cma_nn(self._minim_function, init_point, self.layers, gens, tol=tol, verbose=verbose)
        folder = fold_name(self.name, self.f)
        filename = folder + '/%s_exact.txt' % self.layers
        save_dict(result, folder, filename)

        print(result['fun'])
        return result

    def _minim_function(self, params):
        params = np.asarray(params)
        params = params.reshape((self.layers, 3))
        self.update_parameters(params)
        self.run()
        chi = np.mean((self.final_states - self.function) ** 2)
        return chi


class NN_complex:
    def __init__(self, layers, domain, f, phase):
        self.params = np.random.randn(layers, 3)
        self.domain = domain
        self.function = f(self.domain) * np.exp(1j * 2 * np.pi * phase(self.domain))
        self.final_states = np.zeros(len(self.domain), dtype=complex)
        self.layers = layers
        self.name = 'NN_complex'
        self.f = f
        self.phase = phase

    def update_parameters(self, new_parameters):
        self.params = new_parameters

    def run(self):
        self.final_states = np.zeros_like(self.domain, dtype=complex)
        for i, x in enumerate(self.domain):
            for l in range(self.layers):
                self.final_states[i] += self.params[l, 2] * np.exp(1j * (self.params[l, 0] * x + self.params[l, 1]))

    def find_optimal_parameters(self, init_point=None, gens=100, tol=1e-8, verbose=0):
        if init_point is None:
            init_point = self.params.flatten()

        result = _cma_nn(self._minim_function, init_point, self.layers, gens, tol=tol, verbose=verbose)
        filename = 'results/' + self.name + '/' + self.f.name + '/%s_exact.txt' % self.layers
        with open(filename, 'w') as outfile:
            json.dump(result, outfile)

        return result

    def _minim_function(self, params):
        params = np.asarray(params)
        params = params.reshape((self.layers, 3))
        self.update_parameters(params)
        self.run()
        chi = np.mean(np.abs(self.final_states - self.function) ** 2)
        return (chi,)

class Fourier:
    def __init__(self, layers, domain, f):
        self.domain = domain
        self.function = f(self.domain)# * np.exp(1j * 2 * np.pi * phase(self.domain))
        self.final_states = np.zeros(len(self.domain))
        self.layers = layers
        self.name = 'Fourier'
        self.f = f


    def update_parameters(self, new_parameters):
        self.params = new_parameters

    def run(self):
        self.final_states = np.zeros_like(self.domain)
        x_max = np.max(self.domain)
        x_min = np.min(self.domain)
        for l in range(self.layers):
            sins = np.sin(2 * np.pi * l * self.domain / (x_max - x_min))
            coss = np.cos(2 * np.pi * l * self.domain / (x_max - x_min))
            self.final_states += self.params[l, 0] * coss + self.params[l, 1] * sins

    def find_optimal_parameters(self):
        coeffs = np.zeros((self.layers, 2))
        coeffs[0, 0] = trapz(self.function, self.domain)
        x_max = np.max(self.domain)
        x_min = np.min(self.domain)
        for l in range(1, self.layers):
            sins = 2 * np.sin(2 * np.pi * l * self.domain / (x_max - x_min))
            coss = 2 * np.cos(2 * np.pi * l * self.domain / (x_max - x_min))
            coeffs[l, 0] = np.real(trapz(self.function * coss, self.domain))
            coeffs[l, 1] = np.real(trapz(self.function * sins, self.domain))

        self.params = coeffs / (x_max - x_min)
        self.run()
        result={}
        result['coeffs'] = list(coeffs.flatten())
        result['fun'] = self.approximation()
        print(result['fun'])
        folder = fold_name(self.name, self.f)
        filename = folder + '/%s_exact.txt' % self.layers
        save_dict(result, folder, filename)

        return result

    def approximation(self):
        return np.mean((self.final_states - self.function) ** 2)


class Fourier_complex:
    def __init__(self, layers, domain, f, phase):
        self.params = np.random.randn(layers, 5)
        self.domain = domain
        self.function = f(self.domain) * np.exp(1j * 2 * np.pi * phase(self.domain))
        self.final_states = np.zeros(len(self.domain), dtype=complex)
        self.layers = layers
        self.name = 'Fourier_complex'
        self.f = f
        self.phase = phase


    def update_parameters(self, new_parameters):
        self.params = new_parameters

    def run(self):
        self.final_states = np.zeros_like(self.domain, dtype=complex)
        x_max = np.max(self.domain)
        x_min = np.min(self.domain)
        for l in range(self.layers):
            exp_plus = np.exp(1j * 2 * np.pi * l * self.domain / (x_max - x_min))
            exp_minus = np.exp(-1j * 2 * np.pi * l * self.domain / (x_max - x_min))
            self.final_states += self.params[l, 0] * exp_plus + self.params[l, 1] * exp_minus

    def find_optimal_parameters(self):
        coeffs = np.zeros((self.layers, 2), dtype=complex)
        coeffs[0, 0] = trapz(self.function, self.domain)
        x_max = np.max(self.domain)
        x_min = np.min(self.domain)
        for l in range(1, self.layers):
            exp_plus = np.exp(1j * 2 * np.pi * l * self.domain / (x_max - x_min))
            exp_minus = np.exp(-1j * 2 * np.pi * l * self.domain / (x_max - x_min))
            coeffs[l, 0] = trapz(self.function * exp_plus, self.domain) # There might be some other integration method that arises better values of this calculations
            coeffs[l, 1] = trapz(self.function * exp_minus, self.domain)

        self.params = coeffs / (x_max - x_min)

    def approximation(self):
        return np.mean(np.abs(self.final_states - self.function) ** 2)


def minimize_with_previous_result(algorithm, layers, domain, f, phase=None, gens=100, tol=1e-8):
    try:
        alg = algorithm(layers, domain, f)
    except:
        alg = algorithm(layers, domain, f, phase)
    filename = 'results/' + alg.name + '/' + alg.f.name + '/%s_exact.txt' % (alg.layers - 1)
    with open(filename, 'r') as outfile:
        prev_result = json.load(outfile)

    if not prev_result['success']:
        raise ImportError('Previous result did not converge')

    shape = alg.params.shape

    init_point = np.asfarray(prev_result['x']).reshape((shape[0] - 1, shape[1]))
    init_point = np.concatenate((np.zeros((1, 3)), init_point), axis=0)
    alg.update_parameters(init_point)
    res_q = alg.find_optimal_parameters(gens=gens, tol=tol)
    print(res_q)

def fold_name(appr_name, f):
    folder = 'results/' + appr_name + '/' + f.name
    return folder

def save_dict(result, folder, filename):
    try:
        os.makedirs(folder)
    except:
        pass
    with open(filename, 'w') as outfile:
        json.dump(result, outfile)