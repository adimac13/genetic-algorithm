import itertools
from geneticalgorithm import run, GAParams
from fitness import fitness, repair, fine,trajectory
from dataclasses import replace
import numpy as np
import csv

GRID = {
    "pop_size":        [50, 100, 200],
    "mutation_rate":   [0.05, 0.1, 0.2],
    "sigma":           [1.0, 5.0, 10.0],
    "tournament_size": [2, 3, 5],
}

def run_trials(N:int, params:GAParams, K: int =10) ->dict:
    scores = []
    feasible = 0

    for seed in range(K):
        p = replace(params, seed=seed)
        result = run(N,p,fitness)

        U = repair(result.best_U)
        x = trajectory(U)
        scores.append(np.sum(np.sqrt(U)))
        if fine(U,x) == 0:
            feasible +=1

    scores = np.array(scores)
    return{
        "J_mean": scores.mean(),
        "J_std": scores.std(),
        "J_best": scores.max(),
        "J_worst": scores.min(),
        "feasible": feasible / K,
    }


def grid_search(N:int,grid: dict = GRID, K: int =10) -> list[dict]:
    keys = list(grid.keys())
    results = []

    for values in itertools.product(*grid.values()):
        combo = dict(zip(keys, values))
        params = GAParams(**combo)
        stats = run_trials(N, params, K)
        results.append({"N": N, **combo, **stats})
        print(combo, f"J_mean={stats['J_mean']:.4f}")

    results.sort(key=lambda r: (r["feasible"], r["J_mean"]), reverse=True)
    return results

def save_to_csv(data: list[dict],path:str)->None:
    with open(path,"w",newline="") as f:
        write = csv.DictWriter(f, fieldnames = data[0].keys())
        write.writeheader()
        write.writerows(data)

if __name__ == "__main__":
    for N in (2,4,10, 20, 45):
        results = grid_search(N)
        save_to_csv(results,f"params_test{N}.csv")
        print(f"N={N} the best set: ", results[0])