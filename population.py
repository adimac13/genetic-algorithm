import numpy as np
import random
from constants import N


def generate_population(pop_size: int, N: int, u_max: float) -> np.ndarray:
    return np.random.uniform(low = 0.0,high= u_max,size = (pop_size,N))


def tournament_selection(population: np.ndarray, fitnesses: np.ndarray, tournament_size: int) -> np.ndarray:
    pop_size = population.shape[0]
    idx = np.random.choice( pop_size,tournament_size,replace = False)
    best_fitness_idx = idx[0]
    best_fitness = fitnesses[best_fitness_idx]

    for i in idx[1:]:
        if fitnesses[i] > best_fitness:
            best_fitness = fitnesses[i]
            best_fitness_idx = i

    return population[best_fitness_idx]

def crossover(parent1: np.ndarray, parent2: np.ndarray, alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    child1 = alpha*parent1 +(1-alpha)*parent2
    child2 = alpha*parent2 +(1-alpha)*parent1
    return child1, child2


def gaussian_mutation(individual: np.ndarray, mutation_rate: float, sigma: float, u_max: float) -> np.ndarray:
    subj_copy = individual.copy()
    mask = np.random.rand(len(subj_copy)) < mutation_rate
    noise = np.random.normal(0.0,sigma,size=subj_copy.shape)

    subj_copy[mask] += noise[mask]

    return np.clip(subj_copy,0.0,u_max)



def new_generation(population: np.ndarray, fitnesses: np.ndarray, u_max: float, mutation_rate: float, sigma: float, tournament_size: int) -> np.ndarray:
    pop_size, N = population.shape
    new_population = np.empty_like(population)
    for i in range(pop_size//2):
        parent1 = tournament_selection(population, fitnesses, tournament_size)
        parent2 = tournament_selection(population, fitnesses, tournament_size)

        child1,child2 = crossover(parent1, parent2)

        child1_mutated = gaussian_mutation(child1, mutation_rate, sigma, u_max)
        child2_mutated = gaussian_mutation(child2, mutation_rate, sigma, u_max)

        new_population[2 * i] = child1_mutated
        new_population[2 * i + 1] = child2_mutated

    return new_population
