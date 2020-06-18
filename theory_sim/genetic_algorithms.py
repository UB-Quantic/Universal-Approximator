import numpy as np
import random

from deap import algorithms, base, benchmarks, creator, tools, gp, cma


from theory_sim.classes.Approximant_NN import Approximant_NN
from theory_sim.classes.aux_functions import *
from theory_sim.opt_algorithms import eaGenerateUpdate, eaGenerateUpdate_small
import matplotlib.pyplot as plt

np.random.seed(2)

x = np.linspace(-1,1,101)
f = relu
batch_size=1
ngen = 200
sigma=np.pi
layers=4

q_nn = Approximant_NN(layers, x, f)

creator.create("Fitness", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.Fitness)


toolbox = base.Toolbox()
toolbox.register("attribute", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

q_nn.update_batch_size(batch_size)


# lamb = 4 + 3 * int(np.log(3 * layers))
lamb = 3 * layers * 2

def evaluate(individual, scale=1):
    print(q_nn.layers)
    chi = q_nn._minim_function(individual, sampling=False, noise_sigma=0)
    return chi,

def evaluate_small(individual_small, previous_individual, scale=1):
    chi = q_nn._minim_function(individual_small + previous_individual, sampling=False, noise_sigma=0)
    return chi,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=.5, indpb=1 / (3 * layers))
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("evaluate", evaluate)
toolbox.register("evaluate_small", evaluate_small)


def main_1():
    pop = toolbox.population(n=20)
    crossover_prob = .5
    mutation_prob = 1
    num_gen = 100
    noise=True

    fitnesses = map(toolbox.evaluate, pop, [noise] * len(pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.value = fit



    for g in range(1, num_gen):
        print(g)
        fitnesses_ = [ind.fitness.value for ind in pop]
        print(np.min(fitnesses_), np.max(fitnesses_), np.mean(fitnesses_), np.var(fitnesses_), len(pop))
        offspring = toolbox.select(pop, len(pop))
        # offspring = map(toolbox.clone, offspring)

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob / np.sqrt(g):
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for child in offspring:
            if random.random() < mutation_prob / np.sqrt(g):
                toolbox.mutate(child, indpb=.5 / np.sqrt(g))
                del child.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind, [noise] * len(invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:]=offspring

    return pop


def main_2():
    random.seed()
    MU, LAMBDA = 10, 20
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Se puede cambiar este algoritmo
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                              cxpb=.4, mutpb=.4, ngen=500, stats=stats, halloffame=hof)

    return pop, logbook, hof

def main_3():
    # LAMBDA = 4 + 3 * int(np.log(3 * layers))
    LAMBDA = lamb
    MU = LAMBDA // 2
    #LAMBDA = 3 * layers * 2
    #MU = LAMBDA // 2
    strategy = cma.Strategy([0.0] * 3 * layers, sigma, weights="linear", lambda_=LAMBDA, mu=MU)
    # CMA incluye las mutaciones cada vez más pequeñas
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = eaGenerateUpdate(toolbox, ngen=ngen, stats=stats, halloffame=hof)
    # pop, logbook = eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=ngen, stats=stats, halloffame=hof)
    # pop, logbook = eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, cxpb=.6, mutpb=.3, ngen=ngen, stats=stats, halloffame=hof)

    # Next steps: graduar el landscape
    # Next steps: Regular las probabilidades de mutación y crossover

    return pop, logbook, hof	

def main_4():
    q_nn = Approximant_NN(1, x, f)
    LAMBDA = 4 + 3 * int(np.log(3))
    MU = LAMBDA // 2
    #LAMBDA = 3 * layers * 2
    #MU = LAMBDA // 2
    q_nn = Approximant_NN(1, x, f)
    # LAMBDA = 3 * layers * 2
    # MU = LAMBDA // 2
    strategy = cma.Strategy([0.0] * 3, sigma, weights="linear", lambda_=LAMBDA, mu=MU)
    # CMA incluye las mutaciones cada vez más pequeñas
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = eaGenerateUpdate(toolbox, ngen=ngen, stats=stats, halloffame=hof)
    for layers in range(2, 6):
        q_nn = Approximant_NN(layers, x, f)
        strategy_small = cma.Strategy([0.0] * 3, sigma, weights="linear", lambda_=LAMBDA_small, mu=MU)
        # CMA incluye las mutaciones cada vez más pequeñas
        toolbox.register("generate", strategy_small.generate, creator.Individual)
        toolbox.register("update", strategy_small.update)
        pop_small = toolbox.population(n=MU)
        hof_small = tools.HallOfFame(1)
        stats_small = tools.Statistics(lambda ind: ind.fitness.values)
        stats_small.register("avg", np.mean)
        stats_small.register("std", np.std)
        stats_small.register("min", np.min)
        stats_small.register("max", np.max)
        pop_small, logbook = eaGenerateUpdate_small(toolbox, ngen=ngen, stats=stats_small, halloffame=hof_small)




# TESTING GENETIC ALGORITHMS WITH CMA


# if __name__ == '__main__':
pop, logbook, hof = main_4()
best = pop[0]
print(layers, ' layers')
print(batch_size, ' batch size')
print(hof[0].fitness.values)
params = np.array(hof[0]).reshape((layers, 3))
q_nn.update_parameters(params)
out = q_nn.run(x)
print(np.mean((out - f(x))**2))
plt.plot(x, out)
plt.plot(x, f(x))
plt.show()
'''

