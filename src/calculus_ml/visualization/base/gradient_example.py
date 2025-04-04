import numpy as np
import matplotlib.pyplot as plt

def plot_gradient_example():
    """Vẽ đồ thị minh họa gradient descent với một điểm dữ liệu"""
    # Tạo dữ liệu mẫu
    x_true = 2
    y_true = 400
    
    # Hàm dự đoán và loss
    def predict(x, w):
        return w * x
    
    def loss(w):
        y_pred = predict(x_true, w)
        return 0.5 * (y_pred - y_true)**2
    
    def gradient(w):
        y_pred = predict(x_true, w)
        return (y_pred - y_true) * x_true
    
    # Tạo grid để vẽ
    w = np.linspace(150, 250, 100)
    loss_values = [loss(w_i) for w_i in w]
    
    # Chọn một điểm để minh họa gradient
    w_point = 220
    loss_point = loss(w_point)
    grad_point = gradient(w_point)
    
    # Tạo đường tiếp tuyến
    w_tangent = np.array([w_point - 10, w_point + 10])
    loss_tangent = loss_point + grad_point * (w_tangent - w_point)
    
    # Vẽ đồ thị
    plt.figure(figsize=(10, 6))
    plt.plot(w, loss_values, 'b-', label='Loss function')
    plt.plot(w_point, loss_point, 'ro', label='Current position')
    plt.plot(w_tangent, loss_tangent, 'r--', label='Gradient (slope)')
    
    # Vẽ mũi tên chỉ hướng gradient descent
    plt.arrow(w_point, loss_point, -10, -grad_point*10, 
             head_width=2, head_length=2, fc='g', ec='g',
             label='Gradient descent direction')
    
    plt.title('Gradient Descent Visualization\nExample: Linear Regression with One Point')
    plt.xlabel('Weight (w)')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    
    # Lưu đồ thị
    plt.savefig('images/gradient_descent_example.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_gradient_example() 