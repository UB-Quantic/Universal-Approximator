import numpy as np
import random

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools
from deap import gp

from theory_sim.classes.Approximant_NN import Approximant_NN
from theory_sim.classes.aux_functions import *

layers = 4
x = np.linspace(-1,1,101)
f = relu
q_nn = Approximant_NN(layers, x, f)

creator.create("Fitness", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.Fitness)



toolbox = base.Toolbox()
toolbox.register("attribute", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=layers * 3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

q_nn.update_batch_size(0.1)
q_nn.max_batches = int(np.floor(q_nn.batch_size ** (-1)))
def evaluate(individual):
    chi = q_nn._minim_function(individual, noisy=False)
    return chi,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=.5)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


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

def main():
    random.seed(2)
    MU, LAMBDA = 5, 10
    strategy = cma.Strategy(centroid=[0.0] * 3 * layers, sigma=10.0)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    '''pop, logbook = algorithms.eaGenerateUpdate(# pop,
                                               toolbox, #mu=MU, lambda_=LAMBDA,
                                                ngen=1000,
                                               # cxpb=.4, mutpb=.4, ngen=500,
                                               stats=stats, halloffame=hof)'''

    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                             cxpb=.4, mutpb=.5, ngen=1000, stats=stats, halloffame=hof)

    # Next steps: Regular las probabilidades de mutación y crossover

    return pop, logbook, hof


if __name__ == '__main__':
    pop, logbook, hof = main()
    print(hof[0].fitness.values[0])
    print(evaluate(hof[0]))

