"""
Genetic Algorithm implementation.
"""

from dataclasses import dataclass, field
import numpy as np
from constants import u_max
from algorithms.genetic.population import generate_population, new_generation


@dataclass
class GAParams:
    pop_size: int = 100
    generations: int = 500
    mutation_rate: float = 0.1
    sigma: float = 5.0
    tournament_size: int = 3
    elite_size: int = 2
    seed: int | None = None


@dataclass
class GAResult:
    best_U: np.ndarray
    best_fitness: float
    history_best: list[float] = field(default_factory=list)
    history_mean: list[float] = field(default_factory=list)
    history_worst: list[float] = field(default_factory=list)


def run(N: int, params: GAParams, fitness_fn) -> GAResult:
    if params.seed is not None:
        np.random.seed(params.seed)

    population = generate_population(params.pop_size, N, u_max)
    best_U, best_fit = None, -np.inf
    hist_best, hist_mean, hist_worst = [], [], []

    for _ in range(params.generations):
        fitnesses = np.array([fitness_fn(U) for U in population])

        hist_best.append(fitnesses.max())
        hist_mean.append(fitnesses.mean())
        hist_worst.append(fitnesses.min())

        i = np.argmax(fitnesses)
        if fitnesses[i] > best_fit:
            best_fit, best_U = fitnesses[i], population[i].copy()

        elite = population[np.argsort(fitnesses)[-params.elite_size:]].copy()
        population = new_generation(population, fitnesses, u_max,
                                    params.mutation_rate, params.sigma,
                                    params.tournament_size)
        population[:params.elite_size] = elite

    return GAResult(best_U, best_fit, hist_best, hist_mean, hist_worst)