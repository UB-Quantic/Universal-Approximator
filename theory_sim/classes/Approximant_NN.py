from theory_sim.opt_algorithms import *
import numpy as np
from .aux_functions import *
from scipy.optimize import minimize, differential_evolution, basinhopping



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

    def run_complete(self, noisy=False, samples=10000):
        if not noisy:
            for i, x in enumerate(self.domain):
                self.final_states[i] = state(x, self.params).flatten()
        else:
            self.run_complete()
            probs = np.abs(self.final_states[:, 1]) ** 2
            sampling = np.random.binomial(n=1, p=probs, size=(samples, len(probs)))
            sampling = np.sum(sampling, axis=0) / samples
            self.outcomes = sampling

    def run(self, batch, noisy=False, samples=10000):
        if not noisy:
            final_states = np.zeros((len(batch), 2), dtype=complex)
            for i, x in enumerate(batch):
                final_states[i] = state(x, self.params).flatten()

            return final_states

        else:
            final_states = self.run(batch)
            probs = np.abs(final_states[:, 1]) ** 2
            sampling = np.random.binomial(n=1, p=probs, size=(samples, len(probs)))
            sampling = np.sum(sampling, axis=0) / samples
            return sampling


    def find_optimal_parameters(self, init_point=None, noisy=False, samples=10000, batch_size=1, gens=100, ftol=1e-8, gtol=5e-4, verbose=False):
        if init_point is None:
            init_point = self.params.flatten()
        print(init_point)
        if not noisy:
            result = adam_spsa_optimizer(self._minim_function, init_point, batch_size, ftol=ftol, gtol=gtol)
            # result = minimize(self._minim_function, init_point, args=batch_size, method='powell', options={"disp":verbose})


            # result = _evol('nn', self._minim_function, self.layers, gens, N=100, tol=tol, verbose=verbose)
            # result = _cma('nn', self._minim_function, init_point, self.layers, gens, tol=tol, verbose=verbose)
            # result = minimize(self._minim_function, init_point, args=batch_size, method='powell', options={"disp":verbose})
            # result = adam_optimizer(self._minim_function, init_point, gens=gens)
            #result = basinhopping(self._minim_function, init_point, disp=verbose, minimizer_kwargs={'method':'cobyla', 'args':(batch_size)}, niter_success=3)
            # result = differential_evolution(self._minim_function, [(-np.pi, np.pi)] * len(init_point), disp=verbose)
            print(result)
            print('\n')
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
            print(result)

            folder = fold_name(self.name, self.f)
            filename = folder + '/%s_exact.txt' % self.layers
            # save_dict(result, folder, filename)
            return result


        else:
            # result = _cma_nn(self._minim_function_noisy, init_point, samples, self.layers, gens, tol=tol, verbose=verbose)
            # result = minimize(self._minim_function_noisy, init_point, args=(self.domain, samples), method='l-bfgs-b', options={'disp':verbose, 'ftol':1 / np.sqrt(samples) / len(self.domain)})
            # result = adam_optimizer_noisy(self._minim_function_noisy, init_point, samples)
            result = adam_spsa_optimizer_noisy(self._minim_function_noisy, init_point, batch, samples)
            # result = basinhopping(self._minim_function_noisy, init_point, niter_success=3, minimizer_kwargs={'args':(samples), 'method':'powell', 'options':{'ftol':1 / np.sqrt(samples) / len(self.domain)}}, disp=verbose)
            print(result)
            result['x'] = list(result['x'])
            #del result['message']
            #del result['direc']

        return result

    def _minim_function(self, params, batch_size):
        params = np.asarray(params)
        params = params.reshape((self.layers, 3))
        self.update_parameters(params)
        batch = np.random.choice(self.domain, int(batch_size * len(self.domain)), replace=False)
        batch.sort()

        final_states = self.run(batch) # Debería devolver otras cosas??
        intersect = np.intersect1d(batch, self.domain, return_indices=True)
        #print(batch)
        #print(self.function)  ## HAY ALGÚN PROBLEMA CON LA ORDENACIÓN DE LOS VECTORES, NO SE RESTAN LAS COSAS QUE SE TIENEN QUE RESTAR
        #print(intersect[2])
        #print(self.function[intersect[2]])
        #Hay que reordenar self.function para ser de la misma forma que X
        chi = np.mean((np.abs(final_states[:, 1])**2 - self.function[intersect[2]]) ** 2)
        # print(chi)
        return chi

    def _minim_function_noisy(self, params, batch, samples=10000):
        params = np.asarray(params)
        params = params.reshape((self.layers, 3))
        self.update_parameters(params)
        self.run(X=batch, samples=samples)
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