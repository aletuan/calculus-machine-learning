"""
Định nghĩa các hàm kích hoạt cho mạng neural.
"""

import numpy as np

def sigmoid(x):
    """
    Hàm kích hoạt sigmoid.
    
    Args:
        x (numpy.ndarray): Giá trị đầu vào
        
    Returns:
        numpy.ndarray: Giá trị sau khi áp dụng sigmoid
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Đạo hàm của hàm sigmoid.
    
    Args:
        x (numpy.ndarray): Giá trị đầu vào
        
    Returns:
        numpy.ndarray: Giá trị đạo hàm
    """
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """
    Hàm kích hoạt ReLU (Rectified Linear Unit).
    
    Args:
        x (numpy.ndarray): Giá trị đầu vào
        
    Returns:
        numpy.ndarray: Giá trị đầu ra của hàm ReLU
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Đạo hàm của hàm ReLU.
    
    Args:
        x (numpy.ndarray): Giá trị đầu vào
        
    Returns:
        numpy.ndarray: Đạo hàm của hàm ReLU
    """
    return (x > 0).astype(float)