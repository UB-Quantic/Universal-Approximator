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
