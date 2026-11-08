import numpy as np
from constants import a, x0, N

def evaluate(U: np.ndarray) -> tuple[float, bool, bool]:
    cost = 0
    is_positive = True
    x = x0

    for i in range(N):
        cost += np.sqrt(U[i])
        x = a * x - U[i]
        if x <= 0:
            is_positive = False

    return cost, is_positive, x == x0
