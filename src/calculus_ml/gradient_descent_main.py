import numpy as np
from .cost_function import compute_cost, generate_house_data
from .gradient_descent import gradient_descent, compute_gradient
from .gradient_visualization import plot_gradient_descent, plot_gradient_steps

def main():
    # Tạo dữ liệu
    x_train, y_train = generate_house_data()
    
    # Khởi tạo tham số
    initial_w = 100
    initial_b = 0
    iterations = 1000
    alpha = 0.01
    
    print("Bắt đầu gradient descent:")
    print(f"Giá trị khởi tạo: w = {initial_w}, b = {initial_b}")
    
    # Thực hiện gradient descent
    w_final, b_final, J_hist, p_hist = gradient_descent(
        x_train, y_train, initial_w, initial_b, alpha, iterations, compute_cost)
    
    # Tách lịch sử tham số
    w_hist = [p[0] for p in p_hist]
    b_hist = [p[1] for p in p_hist]
    
    print(f"\nGradient descent hoàn thành:")
    print(f"w = {w_final:.2f}")
    print(f"b = {b_final:.2f}")
    print(f"Cost cuối cùng = {J_hist[-1]:.2f}")
    
    # Vẽ đồ thị quá trình gradient descent
    plot_gradient_descent(x_train, y_train, w_hist, b_hist, J_hist, compute_cost)
    plot_gradient_steps(x_train, y_train, w_hist, b_hist, compute_cost)
    
    print("\nĐã lưu các hình ảnh:")
    print("- gradient_descent_3d.png: Quá trình GD trên bề mặt cost")
    print("- gradient_descent_contour.png: Quá trình GD trên contour plot")
    print("- cost_history.png: Cost function qua các iteration")
    print("- gradient_steps.png: Các bước của gradient descent trên dữ liệu")

if __name__ == "__main__":
    main() 