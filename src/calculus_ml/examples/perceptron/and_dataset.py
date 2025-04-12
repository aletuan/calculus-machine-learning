import numpy as np

def load_and_data():
    """Load the AND gate dataset.
    
    The AND gate is a logical function that returns 1 only when both inputs are 1.
    Truth table:
        x1 | x2 | y
        -----------
        0  | 0  | 0
        0  | 1  | 0
        1  | 0  | 0
        1  | 1  | 1
    
    Returns:
        tuple: (X, y) where
            X (ndarray): Input features of shape (4, 2)
            y (ndarray): Target values of shape (4,)
    """
    # Input features: all possible combinations of 0 and 1
    X = np.array([
        [0, 0],  # Both inputs 0
        [0, 1],  # First input 0, second 1
        [1, 0],  # First input 1, second 0
        [1, 1]   # Both inputs 1
    ])
    
    # Target values: 1 only when both inputs are 1
    y = np.array([0, 0, 0, 1])
    
    return X, y