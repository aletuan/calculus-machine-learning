import numpy as np
import matplotlib.pyplot as plt
import os

def generate_house_data():
    """Tạo dữ liệu mẫu về giá nhà với 1 feature"""
    x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # kích thước (1000 sqft)
    y_train = np.array([300, 500, 700, 900, 1100])  # giá (1000s $)
    return x_train, y_train

def generate_house_data_2d():
    """Tạo dữ liệu mẫu về giá nhà với 2 features"""
    # Feature 1: kích thước (1000 sqft)
    x1_train = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 4.0])
    # Feature 2: số phòng ngủ
    x2_train = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    # Giá nhà (1000s $)
    y_train = np.array([250, 350, 400, 500, 550, 650])
    return x1_train, x2_train, y_train

def predict(x, w, b):
    """Dự đoán giá trị y dựa trên x, w, b cho 1 feature"""
    return w * x + b

def predict_2d(x1, x2, w1, w2, b):
    """Dự đoán giá trị y dựa trên x1, x2, w1, w2, b cho 2 features"""
    return w1 * x1 + w2 * x2 + b

def compute_cost(x, y, w, b):
    """Tính cost function J(w,b) cho 1 feature"""
    m = len(x)
    total_cost = 0
    for i in range(m):
        y_pred = predict(x[i], w, b)
        cost = (y_pred - y[i]) ** 2
        total_cost += cost
    return total_cost / (2 * m)

def compute_cost_2d(x1, x2, y, w1, w2, b):
    """Tính cost function J(w1,w2,b) cho 2 features"""
    m = len(x1)
    total_cost = 0
    for i in range(m):
        y_pred = predict_2d(x1[i], x2[i], w1, w2, b)
        cost = (y_pred - y[i]) ** 2
        total_cost += cost
    return total_cost / (2 * m)

def plot_house_data(x_train, y_train):
    """Vẽ dữ liệu giá nhà"""
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, color='blue', s=100, label='Dữ liệu nhà')
    plt.xlabel('Kích thước (1000 sqft)')
    plt.ylabel('Giá (1000s $)')
    plt.title('Dữ liệu giá nhà')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/house_data.png')
    plt.close()

def plot_different_models(x_train, y_train, models):
    """Vẽ các mô hình dự đoán khác nhau"""
    plt.figure(figsize=(12, 8))
    plt.scatter(x_train, y_train, color='blue', s=100, label='Dữ liệu nhà')

    x_range = np.linspace(0, 6, 100)
    for w, b, cost, color, label in models:
        plt.plot(x_range, predict(x_range, w, b), color, 
                label=f'Mô hình: w={w}, b={b}, Cost={cost:.2f}')

    plt.xlabel('Kích thước (1000 sqft)')
    plt.ylabel('Giá (1000s $)')
    plt.title('Dữ liệu và các mô hình dự đoán khác nhau')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/different_models.png')
    plt.close()

def plot_model_errors(x_train, y_train, w, b, cost):
    """Vẽ chi tiết sai số của mô hình"""
    plt.figure(figsize=(12, 8))
    plt.scatter(x_train, y_train, color='blue', s=100, label='Dữ liệu nhà')
    x_range = np.linspace(0, 6, 100)
    plt.plot(x_range, predict(x_range, w, b), 'r-', label=f'Mô hình: w={w}, b={b}')

    for i in range(len(x_train)):
        x_i = x_train[i]
        y_i = y_train[i]
        y_pred_i = predict(x_i, w, b)
        plt.plot([x_i, x_i], [y_i, y_pred_i], 'k--')
        error = y_pred_i - y_i
        plt.text(x_i + 0.1, (y_i + y_pred_i)/2, f'Sai số: {error:.1f}', 
                 verticalalignment='center')

    plt.xlabel('Kích thước (1000 sqft)')
    plt.ylabel('Giá (1000s $)')
    plt.title(f'Chi tiết sai số của mô hình (Cost = {cost:.2f})')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/model_errors.png')
    plt.close()

def plot_squared_errors(x_train, y_train, w, b, cost):
    """Vẽ trực quan hóa bình phương sai số"""
    plt.figure(figsize=(12, 8))
    plt.scatter(x_train, y_train, color='blue', s=100, label='Dữ liệu nhà')
    x_range = np.linspace(0, 6, 100)
    plt.plot(x_range, predict(x_range, w, b), 'r-', label=f'Mô hình: w={w}, b={b}')

    for i in range(len(x_train)):
        x_i = x_train[i]
        y_i = y_train[i]
        y_pred_i = predict(x_i, w, b)
        error = y_pred_i - y_i
        squared_error = error ** 2
        
        plt.plot([x_i, x_i], [y_i, y_pred_i], 'k--')
        plt.text(x_i + 0.1, (y_i + y_pred_i)/2, 
                 f'Sai số: {error:.1f}\nBình phương: {squared_error:.1f}', 
                 verticalalignment='center')

        sq_size = np.sqrt(squared_error) / 20
        plt.gca().add_patch(plt.Rectangle((x_i - sq_size/2, y_pred_i - sq_size/2), 
                                        sq_size, sq_size, fill=True, color='red', alpha=0.3))

    plt.title(f'Bình phương sai số: J(w,b) = (1/2m) * Σ(f_wb(x) - y)² = {cost:.2f}')
    plt.xlabel('Kích thước (1000 sqft)')
    plt.ylabel('Giá (1000s $)')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/squared_errors.png')
    plt.close()

def plot_cost_contour(x_train, y_train, models):
    """Vẽ đường đồng mức của Cost Function"""
    w_values = np.linspace(100, 300, 20)
    b_values = np.linspace(-200, 200, 20)
    W, B = np.meshgrid(w_values, b_values)

    Z = np.zeros_like(W)
    for i in range(len(w_values)):
        for j in range(len(b_values)):
            Z[j, i] = compute_cost(x_train, y_train, W[j, i], B[j, i])

    plt.figure(figsize=(10, 8))
    plt.contour(W, B, Z, 50, cmap='viridis')
    plt.colorbar(label='Cost J(w,b)')
    plt.xlabel('w')
    plt.ylabel('b')
    plt.title('Đường đồng mức của Cost Function')

    for w, b, cost, color, _ in models:
        plt.plot(w, b, color, markersize=10, label=f'Cost={cost:.2f}')

    plt.grid(True)
    plt.legend()
    plt.savefig('images/cost_contour.png')
    plt.close() 