import numpy as np

def load_data():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])  # không cần reshape vì Keras handle
    return X, y