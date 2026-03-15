import numpy as np

def relu(x):
    # Convert input to NumPy array
    x = np.array(x, dtype=float)
    
    # Apply ReLU element-wise
    return np.maximum(0, x)