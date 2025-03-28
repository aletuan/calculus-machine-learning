import os
import numpy as np
from .vector_operations import (
    create_vectors,
    vector_operations,
    unit_vector_and_angle,
    vector_projection
)
from .vector_visualization import (
    plot_basic_vectors,
    plot_vector_angle,
    plot_vector_projection
)
from .linear_regression import (
    generate_data,
    fit_linear_regression,
    plot_linear_regression
)
from .core.vector import *
from .core.linear_model import *
from .core.cost_function import compute_cost, generate_house_data
from .core.gradient_descent import gradient_descent, compute_gradient
from .visualization.cost_plot import plot_cost_surface, plot_cost_contour
from .visualization.gradient_plot import plot_gradient_descent, plot_gradient_steps

def run_vector_examples():
    """Chạy các ví dụ về vector"""
    print("\n=== Vector Operations Examples ===")
    v1 = [3, 4]
    v2 = [1, 2]
    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print(f"Sum: {add_vectors(v1, v2)}")
    print(f"Dot product: {dot_product(v1, v2)}")
    print(f"Norm of v1: {vector_norm(v1)}")

def run_linear_regression():
    """Chạy ví dụ về linear regression"""
    print("\n=== Linear Regression Example ===")
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    w, b = find_linear_regression(x, y)
    print(f"Found parameters: w = {w:.2f}, b = {b:.2f}")

def run_cost_visualization():
    """Chạy ví dụ về cost function visualization"""
    print("\n=== Cost Function Visualization ===")
    x_train, y_train = generate_house_data()
    w, b = 200, 100
    cost = compute_cost(x_train, y_train, w, b)
    print(f"Cost at w={w}, b={b}: {cost:.2f}")
    plot_cost_surface(x_train, y_train)
    plot_cost_contour(x_train, y_train)

def run_gradient_descent():
    """Chạy ví dụ về gradient descent"""
    print("\n=== Gradient Descent Example ===")
    x_train, y_train = generate_house_data()
    
    # Khởi tạo tham số
    initial_w = 100
    initial_b = 0
    iterations = 1000
    alpha = 0.01
    
    print(f"Initial parameters: w = {initial_w}, b = {initial_b}")
    
    # Thực hiện gradient descent
    w_final, b_final, J_hist, p_hist = gradient_descent(
        x_train, y_train, initial_w, initial_b, alpha, iterations, compute_cost)
    
    # Tách lịch sử tham số
    w_hist = [p[0] for p in p_hist]
    b_hist = [p[1] for p in p_hist]
    
    print(f"Final parameters: w = {w_final:.2f}, b = {b_final:.2f}")
    print(f"Final cost = {J_hist[-1]:.2f}")
    
    # Vẽ đồ thị
    plot_gradient_descent(x_train, y_train, w_hist, b_hist, J_hist, compute_cost)
    plot_gradient_steps(x_train, y_train, w_hist, b_hist, compute_cost)

def main():
    """Hàm main chạy tất cả các ví dụ"""
    print("=== Calculus Machine Learning Examples ===")
    
    # Chạy các ví dụ
    run_vector_examples()
    run_linear_regression()
    run_cost_visualization()
    run_gradient_descent()
    
    print("\nAll examples completed. Check 'images' directory for visualizations.")

if __name__ == "__main__":
    main() 