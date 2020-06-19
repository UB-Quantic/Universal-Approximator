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
