from algorithm import evaluate, optimize_with_scipy
import numpy as np

if __name__ == "__main__":
    U_opt = optimize_with_scipy()
    print("Optimized U:", U_opt)

