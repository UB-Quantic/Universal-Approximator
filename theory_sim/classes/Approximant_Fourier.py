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
