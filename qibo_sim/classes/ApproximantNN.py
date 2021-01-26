from qibo.models import Circuit
from qibo import gates
import numpy as np
from qibo import matrices
import classes.aux_functions as aux
from qibo.hamiltonians import Hamiltonian
import matplotlib.pyplot as plt
import tensorflow as tf
# import tensorflow_probability as tfp

class Approximant:
    def __init__(self, layers, domain, ansatz):
        #Circuit.__init__(self, nqubits)
        self.nqubits = 1
        self.layers = layers
        self.domain = domain

        self.ansatz = ansatz
        self.circuit, self.rotation, self.nparams = globals(
        )[f"ansatz_{ansatz}"](layers)
        self.params = np.random.randn(self.nparams)
        self.save=False

    def set_parameters(self, new_params):
        self.params = new_params

    def get_state(self, x):
        par = self.rotation(self.params, x)
        self.circuit.set_parameters(par)
        state = self.circuit()
        return state

    def cost_function(self, params):
        try:
            params = params.flatten()
        except:
            pass
        self.set_parameters(params)
        cf = 0
        for x, t in zip(self.domain, self.target):
            cf += self.cf_one_point(x, t)
        cf /= len(self.domain)

        return cf

    def cost_function_classical(self, prediction):
        cf = np.mean(np.abs(prediction - self.target)**2)
        return cf

    def minimize(self, method, options={}, compile=True, save=False, **kwargs):
        # WARNING: ONLY CMA AND SCIPY ARE FUNCTIONAL
        self.save = save
        if save:
            self.hist_params = []
            self.hist_chi = []

        if method == 'cma':
            # Genetic optimizer
            import cma
            print(method)
            r = cma.fmin(lambda p: self.cost_function(p).numpy(), self.params, 1, options=options)
            m = {'fun': r[1], 'x': r[0], 'nfev': r[2], 'nit': r[4], 'xmean': r[5], 'stds': r[6]}

            return m

        elif method == 'adam':
            params = adam(self.cost_function, self.derivative_cf, self.params, **kwargs)
            print(params)
            print(self.cost_function(params))
            return params

        elif method=='ego':
            from smt.applications import EGO
            from smt.sampling_methods import FullFactorial
            from smt.problems import Rosenbrock
            from numpy import pi, array
            n_iter = 100
            criterion = 'EI'
            xlimits = array([[-pi, pi]]*len(self.params))
            sampling = FullFactorial(xlimits=xlimits)
            xdoe = sampling(1)
            print(type(xdoe))
            ego = EGO(n_iter=n_iter, criterion=criterion, xdoe=xdoe, xlimits=xlimits)
            x_opt, y_opt , _, x_data, y_data = ego.optimize(fun=lambda p:self.cost_function(p).numpy())
            print(x_opt, y_opt, _)

            m = {'fun': y_opt, 'x': x_opt, 'nfev': 10, 'nit': n_iter, 'xmean': 0, 'stds': 0}


        elif method == 'bayes':
            # Bayesian Optimizer
            from GPyOpt.methods import BayesianOptimization
            import GPyOpt
            from numpy import ones, pi

            bounds = []
            for i in range(len(self.params)):
                bounds.append({'name': 'var_%s'%i, 'type': 'continuous', 'domain': (-3*pi, 3*pi)})

            myBopt2D = BayesianOptimization(self.cost_function, domain=bounds, acquisition_weight=2,
                                            model_type='GP', acquisition_type='EI', normalize_Y=False)

            myBopt2D.run_optimization(500, 50, verbosity=True)

            result = myBopt2D.fx_opt
            parameters = myBopt2D.x_opt
            myBopt2D.plot_convergence()

        elif method == 'SimulatedAnnealing':
            from gradient_free_optimizers import SimulatedAnnealingOptimizer
            import numpy as np
            space_dim = np.array([20])
            init_positions = np.random.rand(len(self.params))

            opt = SimulatedAnnealingOptimizer(init_positions, space_dim, opt_para={})
            for nth_iter in range(len(init_positions), 100):
                pos_new = opt.iterate(nth_iter)
                score_new = self.cost_function(pos_new)  # score must be provided by objective-function
                opt.evaluate(score_new)


        elif method=='OPO':
            import nevergrad as ng
            from numpy import pi
            OPO = ng.optimizers.OnePlusOne(parametrization=len(self.params), budget=10000)
            recommendation = OPO.minimize(lambda p: self.cost_function(p).numpy()[0])

            m = {'fun': self.cost_function(recommendation.value).numpy()[0],
                 'x': recommendation.value}
            return m

        elif method == 'sgd':
            from qibo.tensorflow.gates import TensorflowGate
            circuit = self.circuit(self.domain[0])
            for gate in circuit.queue:
                if not isinstance(gate, TensorflowGate):
                    raise RuntimeError('SGD VQE requires native Tensorflow '
                                       'gates because gradients are not '
                                       'supported in the custom kernels.')

            sgd_options = {"nepochs": 5001,
                           "nmessage": 1000,
                           "optimizer": "Adamax",
                           "learning_rate": 0.5}
            if options is not None:
                sgd_options.update(options)

            # proceed with the training
            from qibo.config import K
            vparams = K.Variable(self.params)
            optimizer = getattr(K.optimizers, sgd_options["optimizer"])(
                learning_rate=sgd_options["learning_rate"])

            def opt_step():
                with K.GradientTape() as tape:
                    l = self.cost_function(vparams)
                grads = tape.gradient(l, [vparams])
                optimizer.apply_gradients(zip(grads, [vparams]))
                return l, vparams

            if compile:
                opt_step = K.function(opt_step)

            l_optimal, params_optimal = 10, self.params
            for e in range(sgd_options["nepochs"]):
                l, vparams = opt_step()
                if l < l_optimal:
                    l_optimal, params_optimal = l, vparams
                if e % sgd_options["nmessage"] == 0:
                    print('ite %d : loss %f' % (e, l.numpy()))

            result = self.cost_function(params_optimal).numpy()
            parameters = params_optimal.numpy()

        elif 'tf' in method:
            from qibo.tensorflow.gates import TensorflowGate
            circuit = self.circuit(self.domain[0])
            for gate in circuit.queue:
                if not isinstance(gate, TensorflowGate):
                    raise RuntimeError('SGD VQE requires native Tensorflow '
                                       'gates because gradients are not '
                                       'supported in the custom kernels.')

            # proceed with the training
            from qibo.config import K
            vparams = K.Variable(self.params)

            def loss_gradient(x):
                return tfp.math.value_and_gradient(lambda x: self.cost_function(x), x)

            if compile:
                loss_gradient = K.function(loss_gradient)

            if 'bfgs' in method:
                def loss_gradient(x):
                    return tfp.math.value_and_gradient(lambda x: self.cost_function(x), x)
                if compile:
                    loss_gradient = K.function(loss_gradient)
                params_optimal = tfp.optimizer.bfgs_minimize(
                    loss_gradient, vparams)
            elif 'lbfgs' in method:
                def loss_gradient(x):
                    return tfp.math.value_and_gradient(lambda x: self.cost_function(x), x)
                if compile:
                    loss_gradient = K.function(loss_gradient)
                params_optimal = tfp.optimizer.lbfgs_minimize(
                    loss_gradient, vparams)
            elif 'nelder_mead' in method:
                params_optimal = tfp.optimizer.nelder_mead_minimize(
                    self.cost_function, initial_vertex=vparams)
            result = params_optimal.objective_value.numpy()
            parameters = params_optimal.position.numpy()

        elif method=='lbfgs_tf':
            from qibo.tensorflow.gates import TensorflowGate
            circuit = self.circuit(self.domain[0])
            for gate in circuit.queue:
                if not isinstance(gate, TensorflowGate):
                    raise RuntimeError('SGD VQE requires native Tensorflow '
                                       'gates because gradients are not '
                                       'supported in the custom kernels.')

            # proceed with the training
            from qibo.config import K
            vparams = K.Variable(self.params)

            def loss_gradient(x):
                return tfp.math.value_and_gradient(lambda x: self.cost_function(x), x)

            if compile:
                loss_gradient = K.function(loss_gradient)

            params_optimal = tfp.optimizer.lbfgs_minimize(
                loss_gradient, vparams)
            result = params_optimal.objective_value.numpy()
            parameters = params_optimal.position.numpy()

            '''elif method == 'l-bfgs-b':
            import numpy as np
            from scipy.optimize import fmin_l_bfgs_b
            m = fmin_l_bfgs_b(lambda p:self.cost_function(p).numpy(), self.params, fprime=self.derivative_cf, args=(),
                        approx_grad=0, bounds=None, m=10, factr=10000000.0,
                        pgtol=1e-05, epsilon=1e-08, iprint=- 1, maxfun=15000, maxiter=15000, disp=None,
                        callback=None, maxls=20)

            print(m)
            return m[0]'''

        else:
            # Newtonian approaches
            import numpy as np
            from scipy.optimize import minimize
            # n = self.hamiltonian.nqubits
            m = minimize(lambda p: self.cost_function(p).numpy(), self.params,
                         method=method, options=options)

            print(m)
            return m


    def run_optimization(self, method, options, compile=True, seed=0):
        np.random.seed(seed)
        folder, trial = self.name_folder()
        result = self.minimize(method, options, compile=compile)
        import pickle
        with open(folder + '/result.pkl', 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
        with open(folder + '/options.pkl', 'wb') as f:
            pickle.dump(options, f, pickle.HIGHEST_PROTOCOL)

        try:
            self.paint_representation_1D(folder + '/plot.pdf')
        except:
            self.paint_representation_2D(folder + '/plot.pdf')
        np.savetxt(folder + '/domain.txt', np.array(self.domain))
        try:
            data = {'method': [method],
                'function':[self.function.name],
                'layers': [self.layers],
                'trial':[trial],
                'seed':[seed],
                'chi2':[result['fun']],
                'ansatz': [self.ansatz],
                'Quantum':[True]}
        except:
            data = {'method': [method],
                'function': [self.f_real.name + '_' + self.f_imag.name],
                'layers': [self.layers],
                'trial': [trial],
                'seed': [seed],
                'chi2': [result['fun']],
                'ansatz': [self.ansatz],
                'Quantum':[True]}

        import pandas as pd
        df = pd.DataFrame(data)
        df.to_csv('summary.csv', mode='a', header=False)

    def run_optimization_classical(self, method, options, seed=0, **kwargs):
        np.random.seed(seed)
        folder, trial = self.name_folder(quantum=False)
        prediction, result = self.classical(self.layers, self.domain, self.target, options, method)
        print(result)
        import pickle
        with open(folder + '/result.pkl', 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
        with open(folder + '/options.pkl', 'wb') as f:
            pickle.dump(options, f, pickle.HIGHEST_PROTOCOL)

        try:
            self.paint_representation_1D_classical(prediction, folder + '/plot.pdf')
        except:
            self.paint_representation_2D_classical(prediction, folder + '/plot.pdf')
        np.savetxt(folder + '/domain.txt', np.array(self.domain))
        try:
            data = {'method': [method],
                    'function': [self.function.name],
                    'layers': [self.layers],
                    'trial': [trial],
                    'seed': [seed],
                    'chi2': [result['fun']],
                    'ansatz': [self.ansatz],
                    'Quantum': [False]}
        except:
            data = {'method': [method],
                    'function': [self.f_real.name + '_' + self.f_imag.name],
                    'layers': [self.layers],
                    'trial': [trial],
                    'seed': [seed],
                    'chi2': [result['fun']],
                    'ansatz': [self.ansatz],
                    'Quantum': [False]}

        import pandas as pd
        df = pd.DataFrame(data)
        df.to_csv('summary.csv', mode='a', header=False)



class Approximant_real(Approximant):
    def __init__(self, layers, domain, ansatz, function):
        self.function = function
        super().__init__(layers, domain, ansatz)
        self.target = self.function(self.domain)
        self.target = 2 * (self.target - np.min(self.target)) / (np.max(self.target) - np.min(self.target)) - 1
        self.H = Hamiltonian(1, matrices._Z)
        self.classical = globals()[f"classical_real_{self.ansatz}"]


    def name_folder(self, quantum=True):
        folder = self.ansatz + '/' + self.function.name + '/%s_layers'%(self.layers)
        if quantum:
            folder = 'quantum/' + folder
        else:
            folder = 'classical/' + folder
        folder = 'results/' + folder
        import os
        try:
            l = os.listdir(folder)
            l = [int(_) for _ in l]
            l.sort()
            trial = int(l[-1]) + 1
        except:
            trial = 0
            os.makedirs(folder)
        fold_name = folder + '/%s' % (trial)
        os.makedirs(fold_name)
        return fold_name, trial

    def cf_one_point(self, x, f):
        state = self.get_state(x)
        o = self.H.expectation(state)
        cf = (o - f) ** 2
        return cf

    def derivative_cf_one_point(self, x, f):
        self.theta_with_x(x)
        derivatives = np.zeros_like(self.params)
        index = 0
        ch_index = 0
        for l in range(self.layers - 1):
            cir_params_ = self.cir_params.copy()
            delta = np.zeros_like(cir_params_)
            delta[ch_index] = np.pi / 2
            self.C.set_parameters(cir_params_ + delta)
            state = self.C()
            z1 = self.H.expectation(state)
            self.C.set_parameters(cir_params_ - delta)
            state = self.C()
            z2 = self.H.expectation(state)
            derivatives[index + 1] = 0.5 * ((z1**2 - z2**2) - 2 * f * (z1 - z2))
            derivatives[index] = x * derivatives[index + 1]
            index += 2
            ch_index += 1

            cir_params_ = self.cir_params.copy()
            delta = np.zeros_like(cir_params_)
            delta[ch_index] = np.pi / 2
            self.C.set_parameters(cir_params_ + delta)
            state = self.C()
            z1 = self.H.expectation(state)
            self.C.set_parameters(cir_params_ - delta)
            state = self.C()
            z2 = self.H.expectation(state)
            derivatives[index] = 0.5 * ((z1**2 - z2**2) - 2 * f * (z1 - z2))
            index += 1
            ch_index += 1

        cir_params_ = self.cir_params.copy()
        delta = np.zeros_like(cir_params_)
        delta[ch_index] = np.pi / 2
        self.C.set_parameters(cir_params_ + delta)
        state = self.C()
        z1 = self.H.expectation(state)
        self.C.set_parameters(cir_params_ - delta)
        state = self.C()
        z2 = self.H.expectation(state)
        derivatives[index + 1] = 0.5 * ((z1**2 - z2**2) - 2 * f * (z1 - z2))
        derivatives[index] = x * derivatives[index + 1]
        index += 2
        ch_index += 1

        return derivatives


    def derivative_cf(self, params):
        try:
            params = params.flatten()
        except:
            pass
        self.set_parameters(params)
        derivatives = np.zeros_like(params)
        for x, t in zip(self.domain, self.target):
            derivatives += self.derivative_cf_one_point(x, t)
        derivatives /= len(self.domain)

        return derivatives

    def paint_representation_1D(self, name):
        fig, axs = plt.subplots()

        axs.plot(self.domain, self.target, color='black', label='Target Function')
        outcomes = np.zeros_like(self.domain)
        for j, x in enumerate(self.domain):
            state = self.get_state(x)
            outcomes[j] = self.H.expectation(state)

        axs.plot(self.domain, outcomes, color='C1',label='Quantum ' + self.ansatz + ' model')

        axs.legend()

        fig.savefig(name)
        plt.close(fig)

    def paint_representation_1D_classical(self, prediction, name):
        fig, axs = plt.subplots()

        axs.plot(self.domain, self.target, color='black', label='Target Function')
        axs.plot(self.domain, prediction,
                 color='C0',label='Classical ' + self.ansatz + ' model')
        axs.legend()

        fig.savefig(name)
        plt.close(fig)

    def paint_representation_2D(self):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if self.num_functions == 1:
            ax = fig.gca(projection='3d')
            print('shape', self.target[0].shape)
            ax.plot_trisurf(self.domain[:, 0], self.domain[:, 1], self.target[:, 0], label='Target Function',  linewidth=0.2, antialiased=True)
            outcomes = np.zeros_like(self.domain)
            for j, x in enumerate(self.domain):
                C = self.circuit(x)
                state = C.execute()
                outcomes[j] = self.hamiltonian[0].expectation(state)

            ax.plot_trisurf(self.domain[:, 0], self.domain[:, 1], outcomes[:, 0] + 0.1,label='Approximation',  linewidth=0.2, antialiased=True)
            #ax.legend()
        else:
            for i in range(self.num_functions):
                ax = fig.add_subplot(1,1,i + 1, projection='3d')
                ax.plot(self.domain, self.functions[i](self.domain).flatten(), color='black')
                outcomes = np.zeros_like(self.domain)
                for j, x in enumerate(self.domain):
                    C = self.circuit(x)
                    state = C.execute()
                    outcomes[j] = self.hamiltonian[i].expectation(state)

                ax.scatter(self.domain[:, 0], self.domain[:, 1], outcomes, color='C0', label=self.measurements[i])
                ax.legend()

        fig.savefig(name)
        plt.close(fig)


    def paint_historical(self, name):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=2)
        axs[0].plot(np.arange(len(self.hist_chi)), self.hist_chi)
        axs[0].set(yscale='log', ylabel=r'$\chi^2$')
        hist_params = np.array(self.hist_params)
        for i in range(len(self.params)):
            axs[1].plot(np.arange(len(self.hist_chi)), hist_params[:, i], 'C%s' % i)

        axs[1].set(ylabel='Parameter', xlabel='Function evaluation')
        fig.suptitle('Historical behaviour', fontsize=16)
        fig.savefig(name)

class Approximant_complex(Approximant):
    def __init__(self, layers, domain, ansatz, real, imag):
        self.f_real = real
        self.f_imag = imag
        # self.function.name = lambda:self.real.name + '_' + self.imag.name
        super().__init__(layers, domain, ansatz)
        self.real = self.f_real(self.domain)
        self.real = 2 * (self.real - np.min(self.real)) / (np.max(self.real) - np.min(self.real)) - 1

        self.imag = self.f_imag(self.domain)
        self.imag = 2 * (self.imag - np.min(self.imag)) / (np.max(self.imag) - np.min(self.imag)) - 1

        self.target = self.real + 1j * self.imag
        self.target /= np.max(np.abs(self.target))

        self.H = [Hamiltonian(1, matrices._X), Hamiltonian(1, matrices._Y)]
        self.classical = globals()[f"classical_complex_{self.ansatz}"]

    def name_folder(self, quantum=True):
        folder = self.ansatz + '/' + self.f_real.name + '_' + self.f_imag.name + '/%s_layers'%(self.layers)
        if quantum:
            folder = 'quantum/' + folder
        else:
            folder = 'classical/' + folder

        folder = 'results/' + folder
        import os
        try:
            l = os.listdir(folder)
            l = [int(_) for _ in l]
            l.sort()
            trial = int(l[-1]) + 1
        except:
            trial = 0
            os.makedirs(folder)
        fold_name = folder + '/%s'%(trial)
        os.makedirs(fold_name)
        return fold_name, trial

    def cf_one_point(self, x, f):
        state = self.get_state(x)
        o = [self.H[0].expectation(state), self.H[1].expectation(state)]
        cf = (o[0] - np.real(f)) ** 2 + (o[1] - np.imag(f)) ** 2
        return cf


    def paint_representation_1D(self, name):
        fig, axs = plt.subplots(nrows=2, figsize=(8,10))

        axs[0].plot(self.domain, np.real(self.target), color='black', label='Target Function')
        outcomes = np.zeros_like(self.domain)
        for j, x in enumerate(self.domain):
            state = self.get_state(x)
            outcomes[j] = self.H[0].expectation(state)

        axs[0].plot(self.domain, outcomes, color='C1',label='Quantum ' + self.ansatz + ' model')
        axs[0].set(title='Real part')
        axs[0].legend()

        axs[1].plot(self.domain, np.imag(self.target), color='black', label='Target Function')
        outcomes = np.zeros_like(self.domain)
        for j, x in enumerate(self.domain):
            state = self.get_state(x)
            outcomes[j] = self.H[1].expectation(state)

        axs[1].plot(self.domain, outcomes, color='C1', label='Quantum ' + self.ansatz + ' model')
        axs[1].set(title='Imag part')
        # axs[1].legend()

        fig.savefig(name)

    def paint_representation_1D_classical(self, prediction, name):
        fig, axs = plt.subplots(nrows=2, figsize=(8,10))

        axs[0].plot(self.domain, np.real(self.target), color='black', label='Target Function')
        axs[0].plot(self.domain, np.real(prediction),
                 color='C0',label='Classical ' + self.ansatz + ' model')
        axs[0].set(title='Real part')
        axs[0].legend()

        axs[1].plot(self.domain, np.imag(self.target), color='black', label='Target Function')
        axs[1].plot(self.domain, np.imag(prediction),
                    color='C0', label='Classical ' + self.ansatz + ' model')
        axs[1].set(title='Imag part')
        # axs[1].legend()

        fig.savefig(name)


class Approximant_real_2D(Approximant):
    def __init__(self, layers, domain, function, ansatz):
        self.function = function
        super().__init__(layers, domain, ansatz)
        self.target = np.array(list(self.function(self.domain)))
        self.target = 2 * (self.target - np.min(self.target)) / (np.max(self.target) - np.min(self.target)) - 1

        self.H = Hamiltonian(1, matrices._Z)
        self.classical = classical_real_Weighted_2D


    def name_folder(self, quantum=True):
        folder = self.ansatz + '/' + self.function.name + '/%s_layers'%(self.layers)
        if quantum:
            folder = 'quantum/' + folder
        else:
            folder = 'classical/' + folder
        folder = 'results/' + folder
        import os
        try:
            l = os.listdir(folder)
            l = [int(_) for _ in l]
            l.sort()
            trial = int(l[-1]) + 1
        except:
            trial = 0
            os.makedirs(folder)
        fold_name = folder + '/%s' % (trial)
        os.makedirs(fold_name)
        return fold_name, trial

    def cf_one_point(self, x, f):
        state = self.get_state(x)
        o = self.H.expectation(state)
        cf = (o - f) ** 2
        return cf

    def paint_representation_2D(self, name):
        fig = plt.figure()
        axs = fig.gca(projection='3d')

        axs.plot_trisurf(self.domain[:, 0], self.domain[:, 1], self.target, color='black', label='Target Function', alpha=0.5)
        outcomes = np.zeros_like(self.target)
        for j, x in enumerate(self.domain):
            state = self.get_state(x)
            outcomes[j] = self.H.expectation(state)

        axs.scatter(self.domain[:, 0], self.domain[:, 1], outcomes, color='C1',label='Quantum ' + self.ansatz + ' model')

        fig.savefig(name)
        plt.close(fig)

    def paint_representation_2D_classical(self, prediction, name):
        fig = plt.figure()
        axs = fig.gca(projection='3d')

        prediction = np.array(list(prediction))

        axs.plot_trisurf(self.domain[:, 0], self.domain[:, 1], self.target, color='black', label='Target Function', alpha=0.5)
        axs.scatter(self.domain[:, 0], self.domain[:, 1], prediction,
                 color='C0',label='Classical ' + self.ansatz + ' model')


        fig.savefig(name)
        #plt.show()
        plt.close(fig)


def adam(loss_function, derivative, init_state, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, T=1000,
         disp=False):
    m = np.zeros_like(init_state)
    v = np.zeros_like(init_state)
    t = 0
    moment_grad = 1
    params = init_state.copy()
    while moment_grad > 0.0001:
        g = derivative(params)

        t += 1
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_ = m / (1 - beta1**t)
        v_ = v / (1 - beta2**t)
        moment_grad = np.linalg.norm(m)
        if disp:
            print('Value of Loss Function at step %s: '%t + str(loss_function(params)), 'Gradient norm: ' + str(moment_grad))

        params -= learning_rate * m_ / (epsilon + v_**.5)
        if t > T:
            break

    return params


from qibo import models

def ansatz_Weighted(layers, qubits=1):
    """
    3 parameters per layer: Ry(wx + a), Rz(b)
    """
    circuit = models.Circuit(qubits)
    circuit.add(gates.H(0))
    for _ in range(layers):
        circuit.add(gates.RZ(0, theta=0))
        circuit.add(gates.RY(0, theta=0))


    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers):
            p[i] = theta[j] + theta[j + 1] * x
            p[i + 1] = theta[j + 2]
            i += 2
            j += 3


        return p

    nparams = 3 * layers
    return circuit, rotation, nparams

def ansatz_Fourier(layers, qubits=1):
    """
    3 parameters per layer: Ry(wx + a), Rz(b)
    """
    circuit = models.Circuit(qubits)
    for _ in range(layers):
        circuit.add(gates.RY(0, theta=0))
        circuit.add(gates.RZ(0, theta=0))
        circuit.add(gates.RY(0, theta=0))
        circuit.add(gates.RZ(0, theta=0))


    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers):
            p[i] = theta[j]
            p[i + 1] = theta[j + 1] + theta[j + 2] * x
            p[i + 2] = theta[j + 3]
            p[i + 3] = theta[j + 4]
            i += 4
            j += 5

        return p

    nparams = 5 * (layers)
    return circuit, rotation, nparams

def ansatz_Weighted_2D(layers, qubits=1):
    """
    3 parameters per layer: Ry(wx + a), Rz(b)
    """
    circuit = models.Circuit(qubits)
    circuit.add(gates.H(0))
    for _ in range(layers):
        circuit.add(gates.RZ(0, theta=0))
        circuit.add(gates.RY(0, theta=0))


    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers):
            p[i] = theta[j] + theta[j + 1: j + 3] @ x
            p[i + 1] = theta[j + 3]
            i += 2
            j += 4


        return p

    nparams = 4 * layers
    return circuit, rotation, nparams


def ansatz_Fourier_2D(layers, qubits=1):
    """
    3 parameters per layer: Ry(wx + a), Rz(b)
    """
    circuit = models.Circuit(qubits)
    for _ in range(layers):
        circuit.add(gates.RY(0, theta=0))
        circuit.add(gates.RZ(0, theta=0))
        circuit.add(gates.RY(0, theta=0))
        circuit.add(gates.RZ(0, theta=0))


    def rotation(theta, x):
        p = circuit.get_parameters()
        i = 0
        j = 0
        for l in range(layers):
            p[i] = theta[j]
            p[i + 1] = theta[j + 1] + theta[j + 2:j + 4] @ x
            p[i + 2] = theta[j + 4]
            p[i + 3] = theta[j + 5]
            i += 4
            j += 6

        return p

    nparams = 6 * (layers)
    return circuit, rotation, nparams

def NN_real(parameters, layers, x, y):
    predict = np.zeros_like(y)
    i = 0
    for l in range(layers):
        predict += parameters[i] * np.cos(parameters[i + 1] * x + parameters[i + 2])
        i += 3

    return predict

def classical_real_Weighted(layers, x, y, options, method):
    nparams = 3 * layers
    def loss(parameters):
        predict = NN_real(parameters, layers, x, y)

        return np.mean((predict - y) ** 2)

    from scipy.optimize import minimize
    result = minimize(loss, x0=np.random.rand(nparams), method=method, options=options)
    prediction = NN_real(result['x'], layers, x, y)

    return prediction, result


def NN_complex(parameters, layers, x, y):
    predict = np.zeros(y.shape, dtype=complex)
    i = 0
    for l in range(layers):
        predict += parameters[i] * np.exp(1j * (parameters[i + 1] * x + parameters[i + 2]))
        i += 3

    return predict

def classical_complex_Weighted(layers, x, y, options, method):
    nparams = 3 * layers
    def loss(parameters):
        predict = NN_complex(parameters, layers, x, y)

        return np.mean(np.abs(predict - y) ** 2)

    from scipy.optimize import minimize
    result = minimize(loss, x0=np.random.rand(nparams), options=options, method=method)
    prediction = NN_complex(result['x'], layers, x, y)

    return prediction, result

def classical_real_Fourier(layers, x, y):
    params = np.zeros((layers + 1, 2))
    period = np.max(x) - np.min(x)
    params[0, 0] = 1 / period * np.trapz(y, x)
    for _ in range(1, layers + 1):
        cos = np.cos(_ * 2 * np.pi / period * x)
        params[_, 0] = 2 / period * np.trapz(y * cos, x)
        sin = np.sin(_ * 2 * np.pi / period * x)
        params[_, 1] = 2 / period * np.trapz(y * sin, x)

    prediction = np.zeros_like(x)
    for i, p in enumerate(params):
        prediction += p[0] * np.cos(i * 2 * np.pi / period * x)
        prediction += p[1] * np.sin(i * 2 * np.pi / period * x)

    result = {'fun': np.mean((prediction - y)**2),
               'x': params}

    return prediction, result


def classical_complex_Fourier(layers, x, y):
    params = np.zeros((layers + 1, 2), dtype=complex)
    period = np.max(x) - np.min(x)
    params[0, 0] = 1 / period * np.trapz(y, x)
    for _ in range(1, layers + 1):
        exp = np.exp(-1j * _ * 2 * np.pi / period * x)
        params[_, 0] = 1 / period * np.trapz(y * exp, x)
        exp = np.exp(1j * _ * 2 * np.pi / period * x)
        params[_, 1] = 1 / period * np.trapz(y * exp, x)

    prediction = np.zeros_like(x, dtype=complex)
    for i, p in enumerate(params):
        prediction += p[0] * np.exp(1j * i * 2 * np.pi / period * x)
        prediction += p[1] * np.exp(-1j * i * 2 * np.pi / period * x)
    result = {'fun': np.mean(np.abs(prediction - y) ** 2),
              'x': params}
    return prediction, result


def NN_real_2D(parameters, layers, x, y):
    predict = np.zeros_like(y)
    for j, x_ in enumerate(x):
        i = 0
        for l in range(layers):
            predict[j] += parameters[i] * np.cos(parameters[i + 1: i + 3] @ x_ + parameters[i + 3])
            i += 4

    return predict

def classical_real_Weighted_2D(layers, x, y, options, method):
    nparams = 4 * layers
    def loss(parameters):
        predict = np.array(list(NN_real_2D(parameters, layers, x, y)))

        return np.mean((predict - y) ** 2)
    from scipy.optimize import minimize
    result = minimize(loss, x0=np.random.rand(nparams), method=method, options=options)
    prediction = NN_real_2D(result['x'],layers, x, y)

    return prediction, result
