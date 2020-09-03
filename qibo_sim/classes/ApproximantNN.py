from qibo.models import Circuit
from qibo import gates
import numpy as np
import classes.aux_functions as aux
from qibo.hamiltonians import Hamiltonian
import matplotlib.pyplot as plt
import tensorflow as tf
# import tensorflow_probability as tfp

np.random.seed(0)

class ApproximantNN:
    def __init__(self, layers, domain, function):
        #Circuit.__init__(self, nqubits)
        self.nqubits = 1
        self.layers = layers
        self.domain = domain
        self.function = function
        self.target = self.function(self.domain)
        self.dimension = domain.shape[1]
        self.hamiltonian = self.create_hamiltonian()
        self.params = np.random.randn(layers * (self.dimension + 2) - 1)
        self.cir_params = np.zeros(layers * 2 - 1)
        self.ansatz()


    def set_parameters(self, new_params):
        self.params = new_params

    def create_hamiltonian(self):
        measur = aux._label_to_hamiltonian('Z')
        h_ = measur[-1]
        self.H = Hamiltonian(1, h_)


    def theta_with_x(self, x):
        index = 0
        ch_index = 0
        if self.dimension == 1:
            for l in range(self.layers - 1):
                self.cir_params[ch_index] = self.params[index : index + self.dimension] * x + self.params[index + self.dimension]
                index += self.dimension + 1
                ch_index += 1
                self.cir_params[ch_index] = self.params[index]
                index += 1
                ch_index += 1

            self.cir_params[ch_index] = self.params[index: index + self.dimension] * x + self.params[index + self.dimension]
            index += self.dimension + 1
            ch_index += 1
            if self.nqubits != 1:
                self.cir_params[ch_index] = self.params[index]
                index += 1
                ch_index += 1
        else:
            for l in range(self.layers - 1):
                    self.cir_params[ch_index] = tf.reduce_sum(self.params[index: index + self.dimension] * x) + \
                                                self.params[index + self.dimension]
                    index += self.dimension + 1
                    ch_index += 1
                    self.cir_params[ch_index] = self.params[index]
                    index += 1
                    ch_index += 1

            self.cir_params[ch_index] = tf.reduce_sum(self.params[index: index + self.dimension] * x) + self.params[
                index + self.dimension]
            index += self.dimension + 1
            ch_index += 1
            self.cir_params[ch_index] = self.params[index]
            index += 1
            ch_index += 1


    def ansatz(self):
        C = Circuit(self.nqubits)
        # x = x.transpose()
        for l in range(self.layers - 1):
            C.add(gates.RY(0, 0))
            C.add(gates.RZ(0, 0))

        C.add(gates.RY(0, 0))

        self.C = C

    def get_state(self, x):
        self.theta_with_x(x)
        self.C.set_parameters(self.cir_params)
        state = self.C()
        return state

    def cost_function_one_point(self, x, f):
        state = self.get_state(x)
        o = self.H.expectation(state)
        cf = (.5 * (1 - o) - f) ** 2

        return cf

    def cost_function(self, params, a=0.0):
        self.set_parameters(params)
        cf = 0
        for x, t in zip(self.domain, self.target):
            cf += self.cost_function_one_point(x, t)
        cf /= len(self.domain)
        return cf

    def minimize(self, method='L-BFGS-B', options=None, compile=True):
        if method == 'cma':
            # Genetic optimizer
            import cma
            r = cma.fmin2(lambda p: self.cost_function(p).numpy(), self.params, 1, restarts=5, bipop=True, options=options)
            result = r[1].result.fbest
            parameters = r[1].result.xbest

        elif method == 'bayes':
            # Bayesian Optimizer
            from GPyOpt.methods import BayesianOptimization
            import GPyOpt
            from numpy import ones, pi

            bounds = []
            for i in range(len(self.params)):
                bounds.append({'name': 'var_%s'%i, 'type': 'continuous', 'domain': (-3*pi, 3*pi)})

            def loss(p):
                f = self.cost_function(p).numpy()
                return f

            myBopt2D = BayesianOptimization(loss, domain=bounds, acquisition_weight=2,
                                            model_type='GP', acquisition_type='EI', normalize_Y=False)

            myBopt2D.run_optimization(5000, 100000000, verbosity=True)

            result = myBopt2D.fx_opt
            parameters = myBopt2D.x_opt
            myBopt2D.plot_convergence()



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

        else:
            # Newtonian approaches
            import numpy as np
            from scipy.optimize import minimize
            # n = self.hamiltonian.nqubits
            m = minimize(lambda p: self.cost_function(p).numpy(), self.params,
                         method=method, options=options)

            print(m)
            result = m.fun
            parameters = m.x

        return result, parameters

    # La minimización puede ser como en el VQE de qibo, la misma estructura es válida, y todos los minimizadores deberían funcionar bien
    # Otro cantar será saber cuál es el minimizador bueno

    def paint_representation_1D(self):
        fig, axs = plt.subplots(nrows=self.num_functions)

        if self.num_functions == 1:
                axs.plot(self.domain, self.functions[0](self.domain), color='black', label='Target Function')
                outcomes = np.zeros_like(self.domain)
                for j, x in enumerate(self.domain):
                    state = self.get_state(x)
                    outcomes[j] = self.hamiltonian[0].expectation(state)

                axs.plot(self.domain, outcomes, color='C0',label='Approximation')
                axs.legend()
        else:
            for i in range(self.num_functions):
                axs[i].plot(self.domain, self.functions[i](self.domain).flatten(), color='black')
                outcomes = np.zeros_like(self.domain)
                for j, x in enumerate(self.domain):
                    state = self.get_state(x)
                    #state = C.execute()
                    outcomes[j] = self.hamiltonian[i].expectation(state)

                axs[i].plot(self.domain, outcomes, label=self.measurements[i])
                axs[i].legend()

        plt.show()

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

        plt.show()
