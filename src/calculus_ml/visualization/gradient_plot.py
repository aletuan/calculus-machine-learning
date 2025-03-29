import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from .cost_plot import ensure_images_dir

def plot_gradient_descent(x, y, w_history, b_history, cost_history, compute_cost, save_as='gradient_descent_3d.png'):
    """
    Vẽ quá trình gradient descent
    
    Args:
        x,y : training data
        w_history, b_history : lịch sử các tham số
        cost_history : lịch sử giá trị cost
        compute_cost : hàm tính cost
        save_as : tên file để lưu (default: 'gradient_descent_3d.png')
    """
    ensure_images_dir()
        
    # Tạo lưới các điểm w,b để vẽ bề mặt
    w = np.linspace(100, 300, 100)
    b = np.linspace(-200, 200, 100)
    W, B = np.meshgrid(w, b)
    
    # Tính cost tại mỗi điểm
    Z = np.zeros_like(W)
    for i in range(len(w)):
        for j in range(len(b)):
            Z[j, i] = compute_cost(x, y, W[j, i], B[j, i])
    
    # Vẽ đồ thị 3D với quá trình gradient descent
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Vẽ bề mặt cost
    surface = ax.plot_surface(W, B, Z, cmap='viridis', alpha=0.7)
    
    # Vẽ đường đi của gradient descent
    w_path = np.array(w_history)
    b_path = np.array(b_history)
    z_path = np.array(cost_history)
    ax.plot3D(w_path, b_path, z_path, 'r.-', linewidth=2, 
              label='Gradient descent path')
    
    # Đánh dấu điểm bắt đầu và kết thúc
    ax.scatter(w_path[0], b_path[0], z_path[0], color='red', s=100, 
              label='Start')
    ax.scatter(w_path[-1], b_path[-1], z_path[-1], color='green', s=100, 
              label='End')
    
    ax.set_xlabel('w (hệ số góc)')
    ax.set_ylabel('b (hệ số tự do)')
    ax.set_zlabel('J(w,b) (cost)')
    plt.title('Quá trình Gradient Descent trên bề mặt Cost Function')
    plt.colorbar(surface, label='Giá trị cost function')
    plt.legend()
    
    plt.savefig(os.path.join('images', save_as), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Vẽ đồ thị 2D của cost function theo số iteration
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, 'b-')
    plt.xlabel('Số iteration')
    plt.ylabel('Cost J(w,b)')
    plt.title('Cost Function qua các iteration')
    plt.grid(True)
    plt.savefig(os.path.join('images', 'cost_history.png'))
    plt.close()
    
    # Vẽ đường đồng mức với quá trình gradient descent
    plt.figure(figsize=(10, 8))
    plt.contour(W, B, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Cost J(w,b)')
    plt.plot(w_path, b_path, 'r.-', linewidth=2, label='Gradient descent path')
    plt.scatter(w_path[0], b_path[0], color='red', s=100, label='Start')
    plt.scatter(w_path[-1], b_path[-1], color='green', s=100, label='End')
    plt.xlabel('w (hệ số góc)')
    plt.ylabel('b (hệ số tự do)')
    plt.title('Quá trình Gradient Descent trên Contour Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('images', 'gradient_descent_contour.png'))
    plt.close()

def plot_gradient_steps(x, y, w_history, b_history, compute_cost, save_as='gradient_descent_steps.png', num_points=5):
    """
    Vẽ các bước của gradient descent trên dữ liệu
    
    Args:
        x,y : training data
        w_history, b_history : lịch sử các tham số
        compute_cost : hàm tính cost
        save_as : tên file để lưu (default: 'gradient_descent_steps.png')
        num_points : số điểm cần vẽ
    """
    ensure_images_dir()
    
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, color='blue', s=100, label='Dữ liệu')
    
    # Chọn một số điểm để vẽ
    indices = np.linspace(0, len(w_history)-1, num_points, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, num_points))
    
    x_range = np.linspace(0, 6, 100)
    for idx, color in zip(indices, colors):
        w = w_history[idx]
        b = b_history[idx]
        cost = compute_cost(x, y, w, b)
        y_pred = w * x_range + b
        plt.plot(x_range, y_pred, color=color, 
                label=f'Iteration {idx}: Cost={cost:.2f}')
    
    plt.xlabel('Kích thước (1000 sqft)')
    plt.ylabel('Giá (1000s $)')
    plt.title('Các bước của Gradient Descent')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('images', save_as))
    plt.close() 