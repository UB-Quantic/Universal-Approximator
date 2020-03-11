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