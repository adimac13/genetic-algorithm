"""
Main script for running and comparing SciPy and Genetic Algorithm optimizations.
"""

from algorithms.scipy.scipy_algorithm import optimize_with_scipy
from algorithms.genetic.genetic_algorithm import run, GAParams
from algorithms.genetic.fitness import fitness, repair, trajectory
from constants import N_values
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    for N in N_values:
        U_opt_scipy = optimize_with_scipy(N)
        J_opt = np.sum(np.sqrt(U_opt_scipy))
        x_scipy = trajectory(U_opt_scipy)


        result = run(N, GAParams(), fitness)
        U_opt_ga = repair(result.best_U)
        history_best_fitness_ga = result.history_best
        history_mean_fitness_ga = result.history_mean
        history_worst_fitness_ga = result.history_worst


        J_ga = np.sum(np.sqrt(U_opt_ga))
        x_ga = trajectory(U_opt_ga)

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), layout='constrained')


        fig.suptitle(f"Optimization Results for N={N}\nJ SciPy={J_opt:.6f}, J GA={J_ga:.6f}")
        axes[0].set_xticks(range(N + 1))
        axes[0].plot(x_scipy,  'b--', label="SciPy")
        axes[0].plot(x_ga,  'r--', label="GA")
        axes[0].legend()
        axes[0].set_ylabel("State")
        axes[0].set_xlabel("Time step")
        axes[0].set_title("State Trajectory")
        axes[0].grid(True)

        axes[1].set_xticks(range(N))
        axes[1].plot(U_opt_scipy,  'b--', label="SciPy")
        axes[1].plot(U_opt_ga,  'r--', label="GA")
        axes[1].legend()
        axes[1].set_ylabel("Control")
        axes[1].set_xlabel("Time step")
        axes[1].set_title("Control Trajectory")
        axes[1].grid(True)

        axes[2].plot(history_best_fitness_ga,   'b--', label="Best Fitness GA")
        axes[2].plot(history_mean_fitness_ga, 'r--', label="Mean Fitness GA")
        # axes[2].plot(history_worst_fitness_ga,'g--', label="Worst Fitness GA")
        axes[2].legend()
        axes[2].set_ylabel("Fitness")
        axes[2].set_xlabel("Generation")
        axes[2].set_title("GA Fitness Evolution")
        axes[2].grid(True)
        plt.show()
