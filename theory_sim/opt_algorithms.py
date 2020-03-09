# Finding the global minimum of a given function
import numpy as np

def f(x):
    return .1 * np.linalg.norm(x) ** 2 + np.sin(x[0] + x[1]) * np.sin(x[0] - x[1])*np.sin(0.1 * np.linalg.norm(x))


import random

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools

def _cma(approximation, function, centroid, layers, gens, N=20, lambda_ = 10, tol=1e-8, verbose=1):
    if 'nn' in approximation.lower():
        p = 3
    elif 'fourier' in approximation.lower():
        p = 5
    #creator.create("fitness", base.Fitness, weights=(-1.0,))
    #creator.create("Individual", list, fitness=creator.fitness)

    toolbox = base.Toolbox()
    toolbox.register("x", np.random.uniform, -np.pi, np.pi)
    try:
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.x, p * layers)
    except:
        creator.create("fitness", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.fitness)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.x, p * layers)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    def evalOneMax(individual):
        return (function(individual),)


    toolbox.register("evaluate", evalOneMax)

    np.random.seed(128)

    strategy = cma.Strategy(centroid=centroid, sigma=np.pi, lambda_=lambda_ * N)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)


    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    record, gen, success = eaGenerateUpdate(toolbox, gens, halloffame=hof, stats=stats, verbose=verbose, tol=tol)

    data = {}
    data['parameters'] = hof[0]
    data['fun'] = hof[0].fitness.values[0]
    data['error'] = record['std']
    data['ngen'] = gen
    data['N'] = N
    data['lambda_'] = lambda_
    data['success'] = success

    return data

def _evol(approximation, function, layers, gens, N=100, lambda_ = 10, tol=1e-8, verbose=True):
    toolbox = base.Toolbox()
    toolbox.register("x", np.random.uniform, -np.pi, np.pi)
    if 'nn' in approximation.lower():
        p = 3
    elif 'fourier' in approximation.lower():
        p = 5
    try:
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.x, p * layers)
    except:
        creator.create("fitness", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.fitness)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.x, p * layers)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    def evalOneMax(individual):
        return (function(individual),)


    toolbox.register("evaluate", evalOneMax)

    np.random.seed(128)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=2.0 / N)
    toolbox.register("select", tools.selTournament, tournsize=3)

    hof = tools.HallOfFame(1)
    pop = toolbox.population(N)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    record, gen, success = eaSimple(pop, toolbox, cxpb=0.4, mutpb=0.5, ngen=gens, stats=stats, halloffame=hof,
                                    verbose=verbose, tol=tol)

    data = {}
    data['x'] = hof[0]
    data['fun'] = hof[0].fitness.values[0]
    data['error'] = record['std']
    data['ngen'] = gen
    data['N'] = N
    data['lambda_'] = lambda_
    data['success'] = success

    return data


def eaGenerateUpdate(toolbox, ngen, halloffame=None, stats=None,
                     verbose=__debug__, tol=1e-8):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    err=1
    gen = 0
    success=True
    while err > tol:
        population = toolbox.generate()
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        toolbox.update(population)

        record = stats.compile(population) if stats is not None else {}
        err = (record['std'])
        logbook.record(gen=gen, nevals=len(population), **record)
        gen += 1
        if verbose:
            print(logbook.stream)
        if gen > ngen:
            success=False
            break

    return record, gen, success


def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, tol=1e-8):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    success=True
    gen=0
    err = 1
    while err > tol:
        offspring = toolbox.select(population, len(population))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)


        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit


        if halloffame is not None:
            halloffame.update(offspring)

        population[:] = offspring

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        gen += 1
        if verbose:
            print(logbook.stream)
        if gen > ngen:
            success = False
            break

    return record, gen, success


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


def adam_spsa_optimizer(function, init_point, batch, a=0.5, b1=0.9, b2=0.999, c=1, gamma=0.5, fmin=0, ftol=1e-8, gtol=1e-3, gens=None):
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
        theta, m, v, cost, conv_rate = adam_spsa_step(function, theta, batch, a, b1, b2, c, gamma, m, v, t)
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

def adam_spsa_step(function, theta, batch, a, b1, b2, c, gamma, m, v, t, epsi=1e-6):
    c_t = c / t ** gamma
    g, cost = adam_spsa_gradient(function, theta, batch, c_t)  # En el paper original de ADAM no hay ninguna referencia a cómo se calcula el gradiente, puede usarse SPSA
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2

    m_ = m / (1 - b1 ** t)
    v_ = v / (1 - b2 ** t)

    theta_new = theta - a * m_ / (np.sqrt(v_) + epsi)
    conv_rate = a * m_ / np.sqrt(v_)
    return theta_new, m, v, cost, conv_rate

def adam_spsa_gradient(function, theta, batch, c_t):
    cost = function(theta)
    displacement = np.random.binomial(1, 0.5, size=theta.shape)
    theta_plus = theta.copy() + c_t * displacement
    theta_minus = theta.copy() - c_t * displacement
    gradient = 1 / 2 / c_t * (function(theta_plus, batch=batch) - function(theta_minus, batch=batch)) * displacement

    return gradient, cost

def adam_spsa_optimizer_noisy(function, init_point, batch, samples, a=0.5, b1=0.9, b2=0.999, c=.1, gamma=0.51, fmin=0, ftol=1e-8, gtol=1e-3, gens=None):
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