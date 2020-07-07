from theory_sim.opt_algorithms import *
import numpy as np
from .aux_functions import *
from scipy.optimize import minimize, differential_evolution, basinhopping



class Approximant_NN:
    # Class to create the mathematical model of our approximant
    def __init__(self, layers, domain, f):
        self.params = np.random.randn(layers, 3) # Parameters defining the classifier
        self.domain = domain # x values
        self.function = f(self.domain) # y values
        self.final_states = np.zeros((len(self.domain), 2), dtype=complex) # Output state of the quantum system
        self.layers = layers # Groups of parameters
        self.f = f # Function to approximate, in the sense of object = function
        self.name = 'q_NN' # Name of the classifier
        self.samples=10000 # Number of measurements done to estimate the outcomes


    def update_parameters(self, new_parameters):
        """
        function to update parameters
        :param new_parameters: new parameters to substitute the old ones
        :return: None, updates parameters
        """
        self.params = new_parameters

    def run_complete(self, noisy=False, samples=10000):
        """
        Function to run the circuit for the whole domain
        :param noisy: Whether the simulation takes wavefunction (False) or sampling (True)
        :param samples: Number of measurements, only taken if noisy=True
        :return: None, updates the outcomes
        """
        if not noisy:
            for i, x in enumerate(self.domain):
                self.final_states[i] = state(x, self.params).flatten()
        else:
            self.run_complete()
            probs = np.abs(self.final_states[:, 1]) ** 2
            sampling = np.random.binomial(n=1, p=probs, size=(samples, len(probs)))
            sampling = np.sum(sampling, axis=0) / samples
            self.outcomes = sampling

    def run(self, batch, noisy=False, samples=None):
        """
        Partial run of the circuit for a batch
        :param batch: subset of the domain where the circuit is ran
        :param noisy: Whether the simulation takes wavefunction (False) or sampling (True)
        :param samples: Number of measurements, only taken if noisy=True
        :return: Outcomes for the batch
        """
        if samples is not None:
            self.samples = samples
        if not noisy:
            final_states = np.zeros((len(batch), 2), dtype=complex)
            for i, x in enumerate(batch):
                final_states[i] = state(x, self.params).flatten()

            return np.abs(final_states[:, 1]) ** 2
        else:
            probs = self.run(batch)
            sampling = np.random.binomial(n=1, p=probs, size=(self.samples, len(probs)))
            sampling = np.sum(sampling, axis=0) / self.samples
            return sampling


    def find_optimal_parameters(self, init_point=None, noisy=False, samples=10000, batch_size=1, gens=100, ftol=1e-8, gtol=5e-4, verbose=False):
        """
        Function to find the optimal parameters
        ################### UNDER CONSTRUCTION #############################
        :param init_point: First guess of parameters
        :param noisy: Whether the simulation takes wavefunction (False) or sampling (True)
        :param samples: Number of measurements, only taken if noisy=True
        :param batch_size: Number between (0, 1), indicates the batch size with respect to the entire domain
        :param gens: Number of iterations of the optimizer, still under construction
        :param ftol: Under construction
        :param gtol: Under construction
        :param verbose: Displays messages
        :return: result of the minimization, depending on the method chosen
        """
        if init_point is None:
            init_point = self.params.flatten()
        self.batch_label=0
        self.batch_size=batch_size
        self.len_partial = int(np.floor(len(self.domain) * batch_size))
        self.max_batches = int(np.floor(batch_size**(-1)))
        self.samples = samples

        # These two first methods are currently working
        # result = self._adam_spsa_optimizer(init_point, ftol=ftol, gtol=gtol, noisy=noisy)
        result = minimize(self._minim_function, init_point, args=noisy, method='powell', options={"disp":verbose})

        # Old code, not fully discarded yet
        # result = basinhopping(self._minim_function, init_point, minimizer_kwargs={'args':noisy, 'options':{'disp':verbose}})
        # result = differential_evolution(self._minim_function, [(-np.pi, np.pi)] * len(init_point), disp=verbose)
        # result = _evol('nn', self._minim_function, self.layers, gens, N=100, tol=tol, verbose=verbose)
        # result = _cma('nn', self._minim_function, init_point, self.layers, gens, tol=tol, verbose=verbose)
        # result = basinhopping(self._minim_function, init_point, disp=verbose, minimizer_kwargs={'method':'cobyla', 'args':(batch_size)}, niter_success=3)
        # result = differential_evolution(self._minim_function, [(-np.pi, np.pi)] * len(init_point), disp=verbose)
        print(result)
        print('\n')
        '''
        result['x'] = list(result['x'])
        try:
            result['jac'] = list(result['jac'])
            del result['message']
            del result['hess_inv']

        except: pass
        try:
            del result['direc']
        except:
            pass
        

        # saver to a dict, not used right now
        folder = fold_name(self.name, self.f)
        filename = folder + '/%s_exact.txt' % self.layers
        # save_dict(result, folder, filename)'''

        print(result)
        return result

    def find_optimal_parameters_by_layers(self, init_point=None, noisy=False, samples=10000, batch_size=1, gens=100, ftol=1e-8, gtol=5e-4, verbose=False):
        """
        Function to find the optimal parameters layer by layer iteratvely, only trial, not successfull yet
        ################### UNDER CONSTRUCTION #############################
        :param init_point: First guess of parameters
        :param noisy: Whether the simulation takes wavefunction (False) or sampling (True)
        :param samples: Number of measurements, only taken if noisy=True
        :param batch_size: Number between (0, 1), indicates the batch size with respect to the entire domain
        :param gens: Number of iterations of the optimizer, still under construction
        :param ftol: Under construction
        :param gtol: Under construction
        :param verbose: Displays messages
        :return: result of the minimization, depending on the method chosen
        """
        if init_point is None:
            init_point = self.params.flatten()
        self.batch_label=0
        self.batch_size=batch_size
        self.len_partial = int(np.floor(len(self.domain) * batch_size))
        self.max_batches = int(np.floor(batch_size**(-1)))
        self.samples = samples
        x = init_point
        print(x)
        for i in range(10):
            for l in range(self.layers - 1, -1, -1):
                result = minimize(self._minim_function_by_layers, x[3*l:3*(l + 1)], args=(l, noisy), method='bfgs')
                print(result)
                p = self.params.copy()
                p[l] = result.x
                self.update_parameters(p)
                print(self.params)



    def _minim_function(self, params, noisy=False):
        """
        Function to compute the chi square of a given set of parameters for the scipy optimizers
        :param params: Parameters
        :param noisy: Wavefunction or sampling
        :return: chi^2
        """
        params = np.asarray(params)
        params = params.reshape((self.layers, 3))
        self.update_parameters(params)
        # print(self.batch_label)
        if self.batch_label == 0:
            s = np.arange(len(self.domain))
            np.random.shuffle(s)
            self.domain = self.domain[s]
            self.function = self.function[s]
            # print(self.domain)
            # print(self.function)
        batch = self.domain[self.batch_label * self.len_partial: (self.batch_label + 1) * self.len_partial]
        batch_function = self.function[self.batch_label * self.len_partial: (self.batch_label + 1) * self.len_partial]
        self.batch_label += 1
        if self.batch_label >= self.max_batches:
            self.batch_label = 0
        outcomes = self.run(batch, noisy)

        # The form of the cost function can be modified in this line
        return np.mean(np.abs(outcomes - batch_function) ** 2)

    def _minim_function_by_layers(self, params, layer, noisy=False):
        """
        Patch to implement layer-by-layer minimization
        :param params: Parameters
        :param layer: layer to optimize
        :param noisy: Wavefunction or sampling
        :return: chi^2
        """
        p = self.params.copy()
        p[3 * layer : 3 * (layer + 1)] = params
        chi = self._minim_function(self.params, noisy=noisy)

        return chi

    def _minim_function_spsa(self, params, batch_label, noisy=False, mixing=False):
        """
        Function to compute the chi square of a given set of parameters with the ADAM-SPSA optimization
        :param params: Parameters
        :param noisy: Wavefunction or sampling
        :return: chi^2
        """
        params = np.asarray(params)
        params = params.reshape((self.layers, 3))
        self.update_parameters(params)
        # print(self.batch_label)
        if mixing:
            s = np.arange(len(self.domain))
            np.random.shuffle(s)
            self.domain = self.domain[s]
            self.function = self.function[s]
            # print(self.domain)
            # print(self.function)
        batch = self.domain[batch_label * self.len_partial: (batch_label + 1) * self.len_partial]
        batch_function = self.function[batch_label * self.len_partial: (batch_label + 1) * self.len_partial]
        outcomes = self.run(batch, noisy)

        return np.mean(np.abs(outcomes - batch_function) ** 2)


    def sampling(self, samples=10000):
        """
        Function to perform sampling
        :param samples: Number of measurements
        :return: Outcomes of the quantum system
        """
        self.run()
        probs = np.abs(self.final_states[:, 1]) ** 2
        sampling = np.random.binomial(n=1, p=probs, size=(samples, len(probs)))
        sampling = np.sum(sampling, axis=0) / samples
        return sampling

    # Below: ADAM + SPSA home made optimizer

    def _adam_spsa_optimizer(self, init_point, a=0.1, b1=0.9, b2=0.999, c=.5, gamma=0.1, fmin=0,
                            ftol=1e-8, gtol=1e-3, gens=None, noisy=False):
        # añadir un máximo de gens si llega el caso
        # añadir opción de verbose
        # Hay un bug en el batch, al principio del gradient
        t = 1
        m = np.zeros_like(init_point)
        v = np.zeros_like(init_point)
        best_theta = init_point.copy()
        theta = init_point.copy()
        best_cost = 1
        batch_label=0
        while best_cost > fmin + ftol:
            theta, m, v, cost, conv_rate = self._adam_spsa_step(theta, batch_label, a, b1, b2, c, gamma, m,
                                                          v, t, noisy)
            t += 1
            batch_label = (batch_label + 1) % self.max_batches
            if t % 100 == 0:
                print(t, cost, np.max(np.abs(conv_rate)))
                print(np.max(np.abs(conv_rate)) < gtol)
            if cost < best_cost:
                best_theta = theta
                best_cost = cost
            if np.max(np.abs(conv_rate)) < gtol:
                break  # checkear condiciones de parada

        data = {}
        data['x'] = best_theta
        data['fun'] = best_cost
        data['error'] = 'Unknown'
        data['ngen'] = t
        data['success'] = 'Unknown'

        return data

    def _adam_spsa_step(self, theta, batch_label, a, b1, b2, c, gamma, m, v, t, noisy, epsi=1e-6):
        c_t = c / t ** gamma
        g, cost = self._adam_spsa_gradient(theta, batch_label, c_t, noisy)  # En el paper original de ADAM no hay ninguna referencia a cómo se calcula el gradiente, puede usarse SPSA

        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2

        m_ = m / (1 - b1 ** t)
        v_ = v / (1 - b2 ** t)

        theta_new = theta - a * m_ / (np.sqrt(v_) + epsi)
        conv_rate = a * m_ / (np.sqrt(v_) + epsi)
        return theta_new, m, v, cost, conv_rate

    def _adam_spsa_gradient(self, theta, batch_label, c_t, noisy):
        if batch_label == 0:
            mix=True
        else:
            mix=False
        cost = self._minim_function_spsa(theta, batch_label, noisy=noisy, mixing=mix)
        displacement = np.random.binomial(1, 0.5, size=theta.shape)
        theta_plus = theta.copy() + c_t * displacement
        theta_minus = theta.copy() - c_t * displacement
        gradient = 1 / 2 / c_t * (
                    self._minim_function_spsa(theta_plus, noisy=noisy, batch_label=batch_label) -
                    self._minim_function_spsa(theta_minus, noisy=noisy,batch_label=batch_label)) * displacement



        return gradient, cost