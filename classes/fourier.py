import numpy as np
from scipy.integrate import trapz

class Fourier:
    def __init__(self, layers, domain, function):
        #Circuit.__init__(self, nqubits)
        self.nqubits = 1
        self.layers = layers
        self.domain = domain
        self.function = function
        self.target = self.function(self.domain)
        self.dimension = domain.shape[1]
        assert self.dimension == 1
        self.domain=domain.flatten()
        self.period = np.max(self.domain) - np.min(self.domain)


    def get_params(self):
        self.params = np.zeros((self.layers + 1, 2), dtype=complex)
        y = self.function(self.domain)
        self.params[0, 0] = 1 / self.period * np.trapz(y, self.domain)
        for _ in range(1, self.layers + 1):
            exp = np.exp(-1j * _ * 2 * np.pi / self.period * self.domain)
            self.params[_, 0] = 1 / self.period * np.trapz(y * exp, self.domain)
            exp = np.exp(1j * _ * 2 * np.pi / self.period * self.domain)
            self.params[_, 1] = 1 / self.period * np.trapz(y * exp, self.domain)


    def predict(self, x):
        y = np.zeros_like(self.domain, dtype=complex)
        for i, p in enumerate(self.params):
            y += p[0] * np.exp(1j * i * 2 * np.pi / self.period * x)
            y += p[1] * np.exp(-1j * i * 2 * np.pi / self.period * x)

        return y

    def cost_function(self, exclude_extrema=True):
        if exclude_extrema:
            y = self.predict(self.domain)[1: -1]
            cf = .5 * np.mean((y - self.function(self.domain)[1: -1]) ** 2)
        else:
            y = self.predict(self.domain)
            cf = .5 * np.mean((y - self.function(self.domain))**2)
        return cf


    def paint_representation_1D(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots()

        axs.plot(self.domain, self.function(self.domain), color='black', label='Target Function')
        axs.plot(self.domain, self.predict(self.domain), color='C0',label='Approximation')
        axs.legend()

        fig.savefig(self.name_folder() + '/%s_layers'%self.layers)

    def name_folder(self):
        folder = 'results/Fourier/' + self.function.name
        import os
        try:
            os.makedirs(folder)
        except:
            pass

        return folder