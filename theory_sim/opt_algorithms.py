# This file is not currently used, but it contains some previously used functions


import numpy as np


import random

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools



from deap.algorithms import varAnd, varOr

def scaling(x):
    # return .5 * (1 + np.tanh(2 * x))
    return 1

def eaGenerateUpdate(toolbox, ngen, halloffame=None, stats=None,
                     verbose=__debug__):
    """This is algorithm implements the ask-tell model proposed in
    [Colette2010]_, where ask is called `generate` and tell is called `update`.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    The algorithm generates the individuals using the :func:`toolbox.generate`
    function and updates the generation method with the :func:`toolbox.update`
    function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The pseudocode goes as follow ::
        for g in range(ngen):
            population = toolbox.generate()
            evaluate(population)
            toolbox.update(population)
    This function expects :meth:`toolbox.generate` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Colette2010] Collette, Y., N. Hansen, G. Pujol, D. Salazar Aponte and
       R. Le Riche (2010). On Object-Oriented Programming of Optimizers -
       Examples in Scilab. In P. Breitkopf and R. F. Coelho, eds.:
       Multidisciplinary Design Optimization in Computational Mechanics,
       Wiley, pp. 527-565;
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    for gen in range(1, ngen + 1):
        # Generate a new population
        population = toolbox.generate()
        # Evaluate the individuals
        '''for ind in population:
            ind.fitness.values = toolbox.evaluate(ind, scale=scaling(1 * gen / ngen))'''
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        # Update the strategy with the evaluated individuals
        toolbox.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

        '''
        
        if record['std'] < 1e-4:
            break
        '''
    return population, logbook


def eaGenerateUpdate_small(toolbox, ngen, halloffame=None, stats=None,
                     verbose=__debug__):
    """This is algorithm implements the ask-tell model proposed in
    [Colette2010]_, where ask is called `generate` and tell is called `update`.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    The algorithm generates the individuals using the :func:`toolbox.generate`
    function and updates the generation method with the :func:`toolbox.update`
    function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The pseudocode goes as follow ::
        for g in range(ngen):
            population = toolbox.generate()
            evaluate(population)
            toolbox.update(population)
    This function expects :meth:`toolbox.generate` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Colette2010] Collette, Y., N. Hansen, G. Pujol, D. Salazar Aponte and
       R. Le Riche (2010). On Object-Oriented Programming of Optimizers -
       Examples in Scilab. In P. Breitkopf and R. F. Coelho, eds.:
       Multidisciplinary Design Optimization in Computational Mechanics,
       Wiley, pp. 527-565;
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    for gen in range(1, ngen + 1):
        # Generate a new population
        population = toolbox.generate()
        # Evaluate the individuals
        '''for ind in population:
            ind.fitness.values = toolbox.evaluate(ind, scale=scaling(1 * gen / ngen))'''
        fitnesses = toolbox.map(toolbox.evaluate_small, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        # Update the strategy with the evaluated individuals
        toolbox.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

        '''

        if record['std'] < 1e-4:
            break
        '''
    return population, logbook

def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::
        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring
    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.
    .. note::
        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.
    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.
    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    #for ind in invalid_ind:
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind, scale=scaling(1 / ngen))

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        #for ind in invalid_ind:
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind, scale=scaling(1 * gen / ngen))


        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.
    The algorithm takes in a population and evolves it in place using the
    :func:`varOr` function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varOr` function. The pseudocode goes as follow ::
        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)
    First, the individuals having an invalid fitness are evaluated. Second,
    the evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by the :func:`varOr` function. The
    offspring are then evaluated and the next generation population is
    selected from both the offspring **and** the population. Finally, when
    *ngen* generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.
    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    '''fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit'''

    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind, scale=scaling(1/ngen))

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind, scale=scaling(1 * gen / ngen))
        '''fitnesses = toolbox.map(toolbox.evaluate, (invalid_ind, 1/ngen))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit'''

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

def adam_optimizer(function, init_point, gens=1000, batch=1, h=0.1, a=0.005, b1=0.9, b2=0.999, fmin=1e-8):
    t = 1
    m = np.zeros_like(init_point)
    v = np.zeros_like(init_point)
    best_theta = init_point.copy()
    theta = init_point.copy()
    best_c = 1
    while t < gens:
        theta, m, v, c = adam_step(function, theta, batch, h, a, b1, b2, m, v, t)
        t+= 1
        if t%20 == 0: print(t, c)
        if c < best_c:
            best_theta = theta
            best_c = c
        if best_c < fmin:
            best_theta = theta
            break  # checkear condiciones de parada


    data = {}
    data['x'] = best_theta
    data['fun'] = best_c
    data['error'] = 'Unknown'
    data['ngen'] = t
    data['success'] = 'Unknown'

    return data

def adam_step(function, theta, batch, h, a, b1, b2, m, v, t, epsi=1e-6):
    g, c = est_grad_2(function, theta, h, batch) # En el paper original de ADAM no hay ninguna referencia a cómo se calcula el gradiente, puede usarse SPSA
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g**2

    m_ = m / (1 - b1**t)
    v_ = v / (1 - b2**t)

    theta_new = theta - a * m_ / (np.sqrt(v_) + epsi)

    return theta_new, m, v, c


def est_grad_1(function, theta, h, batch):
    gradient = np.empty_like(theta)
    theta_plus = theta.copy()
    theta_minus = theta.copy()
    c = function(theta)
    for _ in range(len(theta)):
                theta_plus[_] += h
                theta_minus[_] -= h
                gradient[_] = 1 / 2 / h *(function(theta_plus, batch=batch) - function(theta_minus, batch=batch))
                theta_plus[_] -= h
                theta_minus[_] += h

    return gradient, c


def est_grad_2(function, theta, h, batch):
    c = function(theta, batch)
    displacement = np.random.binomial(1, 0.5, size=theta.shape)
    theta_plus = theta.copy() + displacement
    theta_minus = theta.copy() - displacement
    gradient= 1 / 2 / h *(function(theta_plus, batch=batch) - function(theta_minus, batch=batch)) * displacement

    return gradient, c

def adam_optimizer_noisy(function, init_point, samples, h=0.1, a=0.005, b1=0.9, b2=0.999, tol=1e-8):
    t = 1
    m = np.zeros_like(init_point)
    v = np.zeros_like(init_point)
    best_theta = init_point.copy()
    theta = init_point.copy()
    best_c = 1
    while t < 1000:
        theta, m, v, c = adam_step_noisy(function, theta, samples, h, a, b1, b2, m, v, t)
        t+= 1
        if t%20 == 0: print(t, c)
        if c < best_c:
            best_theta = theta
            best_c = c
        if best_c < tol:
            best_theta = theta
            break

    data = {}
    data['x'] = best_theta
    data['fun'] = best_c
    data['error'] = 'Unknown'
    data['ngen'] = t
    data['success'] = 'Unknown'

    return data

def adam_step_noisy(function, theta, samples, h, a, b1, b2, m, v, t, epsi=1e-6):
    g, c = est_grad_1_noisy(function, theta, samples, h)  # h habría que cambiarlo??
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g**2

    m_ = m / (1 - b1**t)
    v_ = v / (1 - b2**t)

    theta_new = theta - a * m_ / (np.sqrt(v_) + epsi)

    return theta_new, m, v, c


def est_grad_1_noisy(function, theta, samples, h):
    gradient = np.empty_like(theta)
    theta_plus = theta.copy()
    theta_minus = theta.copy()
    c = function(theta, samples)
    for _ in range(len(theta)):
                theta_plus[_] += h
                theta_minus[_] -= h
                gradient[_] = 1 / 2 / h *(function(theta_plus, samples) - function(theta_minus, samples))
                theta_plus[_] -= h
                theta_minus[_] += h

    return gradient, c

def est_grad_2_noisy(function, theta, h, samples, batch):
    c = function(theta, samples, batch)
    displacement = np.random.binomial(1, 0.5, size=theta.shape)
    theta_plus = theta.copy() + displacement
    theta_minus = theta.copy() - displacement
    gradient= 1 / 2 / h *(function(theta_plus, batch=batch) - function(theta_minus, batch=batch)) * displacement

    return gradient, c


def adam_spsa_optimizer(function, init_point, batch_size, domain, a=0.1, b1=0.9, b2=0.999, c=.5, gamma=0.1, fmin=0, ftol=1e-8, gtol=1e-3, gens=None, noisy=False):
    # añadir un máximo de gens si llega el caso
    # añadir opción de verbose
    # Hay un bug en el batch, al principio del gradient
    t = 1
    m = np.zeros_like(init_point)
    v = np.zeros_like(init_point)
    best_theta = init_point.copy()
    theta = init_point.copy()
    best_cost = 1
    while best_cost > fmin + ftol:
        theta, m, v, cost, conv_rate = adam_spsa_step(function, theta, batch_size, domain, a, b1, b2, c, gamma, m, v, t, noisy)
        t += 1
        if t%100 == 0:
            print(t, cost, np.max(np.abs(conv_rate)))
            print(np.max(np.abs(conv_rate)) < gtol)
        if cost < best_cost:
            best_theta = theta
            best_cost = cost
        if np.max(np.abs(conv_rate)) < gtol:

            break  # checkear condiciones de parada


    data = {}
    data['x'] = best_theta
    data['fun'] = best_cost
    data['error'] = 'Unknown'
    data['ngen'] = t
    data['success'] = 'Unknown'

    return data

def adam_spsa_step(function, theta, batch_size, domain, a, b1, b2, c, gamma, m, v, t, noisy, epsi=1e-6):
    c_t = c / t ** gamma
    batch = np.random.choice(domain, int(np.round(batch_size * len(domain))), replace=False)
    g, cost = adam_spsa_gradient(function, theta, batch_label, c_t, noisy)  # En el paper original de ADAM no hay ninguna referencia a cómo se calcula el gradiente, puede usarse SPSA

    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2

    m_ = m / (1 - b1 ** t)
    v_ = v / (1 - b2 ** t)

    theta_new = theta - a * m_ / (np.sqrt(v_) + epsi)
    conv_rate = a * m_ / (np.sqrt(v_) + epsi)
    return theta_new, m, v, cost, conv_rate

def adam_spsa_gradient(function, theta, batch_label, c_t, noisy):
    cost = function(theta, noisy=noisy, batch_label=batch_label)
    displacement = np.random.binomial(1, 0.5, size=theta.shape)
    theta_plus = theta.copy() + c_t * displacement
    theta_minus = theta.copy() - c_t * displacement
    gradient = 1 / 2 / c_t * (function(theta_plus, noisy=noisy, batch_label=batch_label) - function(theta_minus, noisy=noisy, batch_label=batch_label)) * displacement

    return gradient, cost

def adam_spsa_optimizer_noisy(function, init_point, batch, samples, a=0.5, b1=0.9, b2=0.999, c=.5, gamma=0.101, fmin=0, ftol=1e-8, gtol=1e-3, gens=None):
    # añadir un máximo de gens si llega el caso
    # añadir opción de verbose
    t = 1
    m = np.zeros_like(init_point)
    v = np.zeros_like(init_point)
    best_theta = init_point.copy()
    theta = init_point.copy()
    best_cost = 1
    while best_cost > fmin + ftol:
        theta, m, v, cost, conv_rate = adam_spsa_step_noisy(function, theta, batch, samples, a, b1, b2, c, gamma, m, v, t)
        t += 1
        if t%20 == 0:
            print(t, cost, np.max(np.abs(conv_rate)))
            print(np.max(np.abs(conv_rate)) < gtol)
        if cost < best_cost:
            best_theta = theta
            best_cost = cost
        if np.max(np.abs(conv_rate)) < gtol:

            break  # checkear condiciones de parada


    data = {}
    data['x'] = best_theta
    data['fun'] = best_cost
    data['error'] = 'Unknown'
    data['ngen'] = t
    data['success'] = 'Unknown'

    return data

def adam_spsa_step_noisy(function, theta, batch, samples, a, b1, b2, c, gamma, m, v, t, epsi=1e-6):
    c_t = c / t ** gamma
    g, cost = adam_spsa_gradient_noisy(function, theta, batch, samples, c_t)  # En el paper original de ADAM no hay ninguna referencia a cómo se calcula el gradiente, puede usarse SPSA
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2

    m_ = m / (1 - b1 ** t)
    v_ = v / (1 - b2 ** t)

    theta_new = theta - a * m_ / (np.sqrt(v_) + epsi)
    conv_rate = a * m_ / np.sqrt(v_)
    return theta_new, m, v, cost, conv_rate

def adam_spsa_gradient_noisy(function, theta, batch, samples, c_t):
    cost = function(theta, batch, samples)
    displacement = np.random.binomial(1, 0.5, size=theta.shape)
    theta_plus = theta.copy() + c_t * displacement
    theta_minus = theta.copy() - c_t * displacement
    gradient = 1 / 2 / c_t * (function(theta_plus, batch=batch, samples=samples) - function(theta_minus, batch=batch, samples=samples)) * displacement

    return gradient, cost