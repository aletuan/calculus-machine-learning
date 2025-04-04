import numpy as np
import matplotlib.pyplot as plt
import os
from ..core.logistic_regression import predict_logistic, compute_cost_logistic

def ensure_images_dir():
    """Đảm bảo thư mục images tồn tại"""
    if not os.path.exists('images'):
        os.makedirs('images')

def plot_decision_boundary(x1, x2, y, w1, w2, b, save_as='decision_boundary.png'):
    """
    Vẽ đường decision boundary và dữ liệu phân loại
    """
    ensure_images_dir()
    plt.figure(figsize=(10, 8))
    
    # Vẽ dữ liệu
    plt.scatter(x1[y==0], x2[y==0], c='red', label='Trượt', alpha=0.5)
    plt.scatter(x1[y==1], x2[y==1], c='blue', label='Đỗ', alpha=0.5)
    
    # Vẽ decision boundary
    x1_min, x1_max = x1.min()-1, x1.max()+1
    x2_min, x2_max = x2.min()-1, x2.max()+1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                          np.linspace(x2_min, x2_max, 100))
    Z = predict_logistic(xx1.ravel(), xx2.ravel(), w1, w2, b)
    Z = Z.reshape(xx1.shape)
    
    plt.contour(xx1, xx2, Z, levels=[0.5], colors='g', label='Decision Boundary')
    plt.xlabel('Điểm thi')
    plt.ylabel('GPA')
    plt.title('Phân loại kết quả tuyển sinh')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('images', save_as))
    plt.close()

def plot_cost_surface_logistic(x1, x2, y, w1_range, w2_range, b,
                             save_as='logistic_cost_3d.png'):
    """
    Vẽ bề mặt cost function cho logistic regression
    """
    ensure_images_dir()
    w1 = np.linspace(w1_range[0], w1_range[1], 100)
    w2 = np.linspace(w2_range[0], w2_range[1], 100)
    W1, W2 = np.meshgrid(w1, w2)
    
    Z = np.zeros_like(W1)
    for i in range(len(w1)):
        for j in range(len(w2)):
            Z[j,i] = compute_cost_logistic(x1, x2, y, W1[j,i], W2[j,i], b)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(W1, W2, Z, cmap='viridis')
    
    ax.set_xlabel('w₁')
    ax.set_ylabel('w₂')
    ax.set_zlabel('J(w₁,w₂,b)')
    plt.title('Cost Function cho Logistic Regression')
    plt.colorbar(surface)
    plt.savefig(os.path.join('images', save_as))
    plt.close()

def plot_gradient_descent_logistic(x1, x2, y, w1_history, w2_history, b_history, J_history,
                                 save_as='logistic_gradient_descent.png'):
    """
    Vẽ quá trình gradient descent cho logistic regression
    """
    ensure_images_dir()
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Decision boundary qua các iteration
    plt.subplot(121)
    plt.scatter(x1[y==0], x2[y==0], c='red', label='Trượt', alpha=0.5)
    plt.scatter(x1[y==1], x2[y==1], c='blue', label='Đỗ', alpha=0.5)
    
    # Vẽ decision boundary tại một số iteration
    iterations = [0, len(w1_history)//2, -1]
    colors = ['r--', 'g--', 'b-']
    labels = ['Start', 'Middle', 'Final']
    
    x1_min, x1_max = x1.min()-1, x1.max()+1
    x2_min, x2_max = x2.min()-1, x2.max()+1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                          np.linspace(x2_min, x2_max, 100))
    
    for i, iter_idx in enumerate(iterations):
        w1 = w1_history[iter_idx]
        w2 = w2_history[iter_idx]
        b = b_history[iter_idx]
        Z = predict_logistic(xx1.ravel(), xx2.ravel(), w1, w2, b)
        Z = Z.reshape(xx1.shape)
        plt.contour(xx1, xx2, Z, levels=[0.5], colors=colors[i][0], 
                   linestyles=colors[i][1:], label=labels[i])
    
    plt.xlabel('Điểm thi')
    plt.ylabel('GPA')
    plt.title('Decision Boundary qua các iteration')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Cost history
    plt.subplot(122)
    plt.plot(J_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost J(w₁,w₂,b)')
    plt.title('Cost Function qua các iteration')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join('images', save_as))
    plt.close() 