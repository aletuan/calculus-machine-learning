import numpy as np
import matplotlib.pyplot as plt

def generate_data():
    """Tạo dữ liệu mẫu cho hồi quy tuyến tính"""
    np.random.seed(42)
    x = np.array([1, 2, 3, 4, 5])
    # y = 2x + 1 + noise
    y = 2 * x + 1 + np.random.randn(5) * 0.5
    return x, y

def find_linear_regression(x, y):
    """Tính toán các tham số của hồi quy tuyến tính"""
    # Chuyển x thành ma trận thiết kế (thêm cột 1 cho bias)
    X = np.column_stack((np.ones_like(x), x))
    
    # Giải phương trình tuyến tính để tìm w và b
    # (X^T * X)^(-1) * X^T * y
    params = np.linalg.inv(X.T @ X) @ X.T @ y
    
    # Lấy ra hệ số
    b = params[0]  # bias
    w = params[1]  # weight
    
    return w, b

def plot_linear_regression(x, y, w, b):
    """Vẽ dữ liệu và đường hồi quy"""
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Dữ liệu')
    plt.plot(x, w*x + b, 'r-', label=f'y = {w:.4f}x + {b:.4f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/linear_regression.png')
    plt.close() 