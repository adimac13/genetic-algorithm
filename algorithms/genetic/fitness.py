"""
Fitness functions and trajectory handling for the genetic algorithm.
"""

import numpy as np
from constants import a, x0, u_max

def trajectory(U: np.ndarray) -> np.ndarray:
    x = np.empty(len(U) + 1)
    x[0] = x0
    for k in range(len(U)):
        x[k + 1] = a * x[k] - U[k]
    return x


def repair(U: np.ndarray) -> np.ndarray:
    U=U.copy()
    x = trajectory(U[:-1])
    U[-1]= a*x[-1]-x0
    U[-1] = np.clip(U[-1],0.0,u_max)
    return U

def fine(U:np.ndarray,x:np.ndarray, eps:float=1e-6)-> float:
    x = np.array(x)
    counter = 0.0
    err = abs(x[-1] - x0)
    if err > eps:
        counter+=err

    counter += np.sum(np.maximum(0.0,-x))
    counter += np.sum(np.maximum(0.0, -U))
    counter += np.sum(np.maximum(0.0, U - u_max))
    return counter

def fitness(U:np.ndarray, lamb: float =1e3, use_repair: bool = True) -> float:
    if use_repair:
        U = repair(U)

    x = trajectory(U)
    J = np.sum(np.sqrt(U))
    return J - lamb * fine(U,x)
