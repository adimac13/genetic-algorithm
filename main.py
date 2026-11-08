from algorithm import evaluate
import numpy as np

if __name__ == "__main__":
    U = np.array([1.0, 2.0, -2.0])
    cost, is_positive, is_x_equal_x0 = evaluate(U)
    print("Cost:", cost)
    print("Is positive:", is_positive)
    print("Is x equal to x0:", is_x_equal_x0)
