import numpy as np
from constants import a, x0, N, u_max
from scipy.optimize import minimize

def evaluate(U: np.ndarray) -> tuple[float, bool, bool]:
    cost = 0
    x = [x0]

    for i in range(N):
        cost += np.sqrt(U[i])
        x.append(a * x[-1] - U[i])

    return cost, x

def optimize_with_scipy() -> np.ndarray:
    def fun(U: np.ndarray) -> float:
        cost, _ = evaluate(U)
        return (-1) * cost

    U0 = np.ones(N)
    result = minimize(fun, U0, bounds = [(0, u_max) for _ in range(N)], constraints=
                      [
                          {'type': 'eq', 'fun': lambda U: evaluate(U)[1][-1] - x0 },
                          {'type': 'ineq', 'fun': lambda U: min(evaluate(U)[1]) }
                          ], 
                      method='SLSQP', tol=1e-6)
    return result.x