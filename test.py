import numpy as np
from sklearn.datasets import load_iris

y = np.array([0, 0, 4, 2, 2, 2, 1, 1, 0])
n_y = y[y > 2]
print(n_y)