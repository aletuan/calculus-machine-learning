"""
Tạo dữ liệu cho hàm logic AND.
Hàm AND trả về 1 nếu cả hai đầu vào đều là 1, ngược lại trả về 0.
"""

import numpy as np

def load_and_data():
    """
    Tạo dữ liệu cho hàm AND.
    
    Returns:
        X (numpy.ndarray): Ma trận đầu vào với các cặp giá trị (0,0), (0,1), (1,0), (1,1)
        y (numpy.ndarray): Vector nhãn tương ứng với kết quả của hàm AND
    """
    # Tạo tất cả các tổ hợp có thể của đầu vào
    X = np.array([
        [0, 0],  # 0 AND 0 = 0
        [0, 1],  # 0 AND 1 = 0
        [1, 0],  # 1 AND 0 = 0
        [1, 1]   # 1 AND 1 = 1
    ])
    
    # Tạo nhãn tương ứng với kết quả của hàm AND
    y = np.array([0, 0, 0, 1])
    
    return X, y