import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_cost_surface(x, y, w_range=(100, 300), b_range=(-200, 200)):
    """
    Vẽ bề mặt cost function trong không gian 3D
    """
    if not os.path.exists('images'):
        os.makedirs('images')
        
    # Tạo lưới các điểm w,b
    w = np.linspace(w_range[0], w_range[1], 100)
    b = np.linspace(b_range[0], b_range[1], 100)
    W, B = np.meshgrid(w, b)
    
    # Tính cost tại mỗi điểm
    Z = np.zeros_like(W)
    for i in range(len(w)):
        for j in range(len(b)):
            Z[j, i] = compute_cost(x, y, W[j, i], B[j, i])
    
    # Vẽ đồ thị 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(W, B, Z, cmap='viridis')
    
    ax.set_xlabel('w (hệ số góc)')
    ax.set_ylabel('b (hệ số tự do)')
    ax.set_zlabel('J(w,b) (cost)')
    plt.title('Cost Function J(w,b) trong Không Gian 3D')
    plt.colorbar(surface)
    
    plt.savefig('images/cost_3d.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_cost_contour(x, y, w_range=(100, 300), b_range=(-200, 200)):
    """
    Vẽ đường đồng mức của cost function
    """
    if not os.path.exists('images'):
        os.makedirs('images')
        
    # Tạo lưới các điểm w,b
    w = np.linspace(w_range[0], w_range[1], 100)
    b = np.linspace(b_range[0], b_range[1], 100)
    W, B = np.meshgrid(w, b)
    
    # Tính cost tại mỗi điểm
    Z = np.zeros_like(W)
    for i in range(len(w)):
        for j in range(len(b)):
            Z[j, i] = compute_cost(x, y, W[j, i], B[j, i])
    
    # Vẽ contour plot
    plt.figure(figsize=(10, 8))
    plt.contour(W, B, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Cost J(w,b)')
    plt.xlabel('w (hệ số góc)')
    plt.ylabel('b (hệ số tự do)')
    plt.title('Đường Đồng Mức của Cost Function')
    plt.grid(True)
    
    plt.savefig('images/cost_contour.png')
    plt.close()

def plot_model_errors(x, y, w, b):
    """
    Vẽ chi tiết sai số của mô hình
    """
    if not os.path.exists('images'):
        os.makedirs('images')
        
    plt.figure(figsize=(10, 6))
    
    # Vẽ dữ liệu thực tế
    plt.scatter(x, y, color='blue', label='Dữ liệu thực tế')
    
    # Vẽ đường dự đoán
    x_line = np.linspace(min(x), max(x), 100)
    y_line = w * x_line + b
    plt.plot(x_line, y_line, color='red', label='Mô hình dự đoán')
    
    # Vẽ các đường sai số
    for i in range(len(x)):
        y_pred = w * x[i] + b
        plt.plot([x[i], x[i]], [y[i], y_pred], 'g--', alpha=0.5)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Chi Tiết Sai Số của Mô Hình')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('images/model_errors.png')
    plt.close()

def plot_squared_errors(x, y, w, b):
    """
    Vẽ bình phương sai số
    """
    if not os.path.exists('images'):
        os.makedirs('images')
        
    plt.figure(figsize=(10, 6))
    
    # Vẽ dữ liệu và mô hình
    plt.scatter(x, y, color='blue', label='Dữ liệu thực tế')
    x_line = np.linspace(min(x), max(x), 100)
    y_line = w * x_line + b
    plt.plot(x_line, y_line, color='red', label='Mô hình dự đoán')
    
    # Vẽ các hình vuông thể hiện bình phương sai số
    for i in range(len(x)):
        y_pred = w * x[i] + b
        error = y_pred - y[i]
        
        # Vẽ hình vuông
        plt.fill([x[i]-abs(error)/2, x[i]+abs(error)/2, 
                 x[i]+abs(error)/2, x[i]-abs(error)/2],
                [y[i], y[i], y_pred, y_pred],
                'r', alpha=0.2)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trực Quan Hóa Bình Phương Sai Số')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('images/squared_errors.png')
    plt.close() 