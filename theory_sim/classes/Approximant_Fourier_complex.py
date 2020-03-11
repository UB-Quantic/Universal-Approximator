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
