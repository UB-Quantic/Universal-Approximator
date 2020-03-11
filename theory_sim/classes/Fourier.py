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