from algorithm import optimize_with_scipy
from geneticalgorithm import run, GAParams
from fitness import fitness, repair, trajectory
from constants import N
import numpy as np

if __name__ == "__main__":
    U_opt = optimize_with_scipy()
    J_opt = np.sum(np.sqrt(U_opt))

    result = run(N, GAParams(), fitness)
    U_ga = repair(result.best_U)
    J_ga = np.sum(np.sqrt(U_ga))
    print(f"N={N}")
    print(f"SciPy: J={J_opt:.6f}")
    print(f"AG:    J={J_ga:.6f}, x_N={trajectory(U_ga)[-1]:.6f}")
    print(f"Różnica względna: {abs(J_ga - J_opt) / J_opt:.2e}")

