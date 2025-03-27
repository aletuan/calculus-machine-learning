import numpy as np
import matplotlib.pyplot as plt

# 1. Tạo dữ liệu đơn giản - chỉ 5 mẫu
x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # kích thước (1000 sqft)
y_train = np.array([300, 500, 700, 900, 1100])  # giá (1000s $)

# 2. Hiển thị dữ liệu
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='blue', s=100, label='Dữ liệu nhà')
plt.xlabel('Kích thước (1000 sqft)')
plt.ylabel('Giá (1000s $)')
plt.title('Dữ liệu giá nhà')
plt.grid(True)
plt.legend()
plt.show()

# 3. Định nghĩa hàm để tính dự đoán
def predict(x, w, b):
    """
    Dự đoán giá trị y dựa trên x, w, b
    y = w*x + b
    """
    return w * x + b

# 4. Định nghĩa hàm tính cost function
def compute_cost(x, y, w, b):
    """
    Tính cost function J(w,b)
    J(w,b) = (1/2m) * Σ(f(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
    
    Args:
        x: Dữ liệu đầu vào
        y: Dữ liệu đầu ra thực tế
        w, b: Tham số của mô hình
    
    Return:
        cost: Giá trị của cost function
    """
    m = len(x)  # số lượng mẫu
    
    # Tính tổng bình phương sai số
    total_cost = 0
    for i in range(m):
        # Dự đoán giá trị
        y_pred = predict(x[i], w, b)
        
        # Tính bình phương sai số
        cost = (y_pred - y[i]) ** 2
        
        # Cộng vào tổng
        total_cost += cost
    
    # Tính trung bình và chia cho 2
    return total_cost / (2 * m)

# 5. Tính cost function cho một số giá trị khác nhau của w và b
print("Tính cost function với các giá trị w, b khác nhau:")

# Thử nghiệm 1: w=100, b=100
w1, b1 = 100, 100
cost1 = compute_cost(x_train, y_train, w1, b1)
print(f"w = {w1}, b = {b1} => Cost = {cost1:.2f}")

# Thử nghiệm 2: w=200, b=100
w2, b2 = 200, 100
cost2 = compute_cost(x_train, y_train, w2, b2)
print(f"w = {w2}, b = {b2} => Cost = {cost2:.2f}")

# Thử nghiệm 3: w=200, b=0
w3, b3 = 200, 0
cost3 = compute_cost(x_train, y_train, w3, b3)
print(f"w = {w3}, b = {b3} => Cost = {cost3:.2f}")

# Thử nghiệm 4: w=200, b=-100
w4, b4 = 200, -100
cost4 = compute_cost(x_train, y_train, w4, b4)
print(f"w = {w4}, b = {b4} => Cost = {cost4:.2f}")

# 6. Hiển thị giá trị dự đoán của các mô hình khác nhau
plt.figure(figsize=(12, 8))
plt.scatter(x_train, y_train, color='blue', s=100, label='Dữ liệu nhà')

# Vẽ đường hồi quy cho các mô hình
x_range = np.linspace(0, 6, 100)
plt.plot(x_range, predict(x_range, w1, b1), 'r-', label=f'Mô hình 1: w={w1}, b={b1}, Cost={cost1:.2f}')
plt.plot(x_range, predict(x_range, w2, b2), 'g-', label=f'Mô hình 2: w={w2}, b={b2}, Cost={cost2:.2f}')
plt.plot(x_range, predict(x_range, w3, b3), 'y-', label=f'Mô hình 3: w={w3}, b={b3}, Cost={cost3:.2f}')
plt.plot(x_range, predict(x_range, w4, b4), 'm-', label=f'Mô hình 4: w={w4}, b={b4}, Cost={cost4:.2f}')

plt.xlabel('Kích thước (1000 sqft)')
plt.ylabel('Giá (1000s $)')
plt.title('Dữ liệu và các mô hình dự đoán khác nhau')
plt.grid(True)
plt.legend()
plt.show()

# 7. Minh họa chi tiết cách tính cost function với Mô hình 1
w, b = w1, b1  # Sử dụng mô hình 1

plt.figure(figsize=(12, 8))
plt.scatter(x_train, y_train, color='blue', s=100, label='Dữ liệu nhà')
plt.plot(x_range, predict(x_range, w, b), 'r-', label=f'Mô hình: w={w}, b={b}')

# Vẽ sai số của mỗi điểm dữ liệu
for i in range(len(x_train)):
    x_i = x_train[i]
    y_i = y_train[i]
    y_pred_i = predict(x_i, w, b)
    
    # Vẽ đường thẳng từ điểm thực tế đến điểm dự đoán
    plt.plot([x_i, x_i], [y_i, y_pred_i], 'k--')
    
    # Hiển thị sai số
    error = y_pred_i - y_i
    plt.text(x_i + 0.1, (y_i + y_pred_i)/2, f'Sai số: {error:.1f}', 
             verticalalignment='center')

plt.xlabel('Kích thước (1000 sqft)')
plt.ylabel('Giá (1000s $)')
plt.title(f'Chi tiết sai số của mô hình (Cost = {cost1:.2f})')
plt.grid(True)
plt.legend()
plt.show()

# 8. Trực quan hóa bình phương sai số
plt.figure(figsize=(12, 8))
plt.scatter(x_train, y_train, color='blue', s=100, label='Dữ liệu nhà')
plt.plot(x_range, predict(x_range, w, b), 'r-', label=f'Mô hình: w={w}, b={b}')

# Vẽ bình phương sai số ở mỗi điểm
for i in range(len(x_train)):
    x_i = x_train[i]
    y_i = y_train[i]
    y_pred_i = predict(x_i, w, b)
    
    # Tính bình phương sai số
    error = y_pred_i - y_i
    squared_error = error ** 2
    
    # Vẽ đường thẳng từ điểm thực tế đến điểm dự đoán
    plt.plot([x_i, x_i], [y_i, y_pred_i], 'k--')
    
    # Hiển thị bình phương sai số
    plt.text(x_i + 0.1, (y_i + y_pred_i)/2, 
             f'Sai số: {error:.1f}\nBình phương: {squared_error:.1f}', 
             verticalalignment='center')

    # Vẽ hình vuông để minh họa bình phương sai số
    sq_size = np.sqrt(squared_error) / 20  # Chia để hình vuông không quá lớn
    plt.gca().add_patch(plt.Rectangle((x_i - sq_size/2, y_pred_i - sq_size/2), 
                                      sq_size, sq_size, fill=True, color='red', alpha=0.3))

# Tính cost function
cost = compute_cost(x_train, y_train, w, b)
plt.title(f'Bình phương sai số: J(w,b) = (1/2m) * Σ(f_wb(x) - y)² = {cost:.2f}')
plt.xlabel('Kích thước (1000 sqft)')
plt.ylabel('Giá (1000s $)')
plt.grid(True)
plt.legend()
plt.show()

# 9. Tìm giá trị tối ưu cho w và b
# Tạo lưới các giá trị w và b
w_values = np.linspace(100, 300, 20)
b_values = np.linspace(-200, 200, 20)
W, B = np.meshgrid(w_values, b_values)

# Tính cost function cho mỗi cặp giá trị (w, b)
Z = np.zeros_like(W)
for i in range(len(w_values)):
    for j in range(len(b_values)):
        Z[j, i] = compute_cost(x_train, y_train, W[j, i], B[j, i])

# Vẽ contour plot
plt.figure(figsize=(10, 8))
plt.contour(W, B, Z, 50, cmap='viridis')
plt.colorbar(label='Cost J(w,b)')
plt.xlabel('w')
plt.ylabel('b')
plt.title('Đường đồng mức của Cost Function')

# Đánh dấu các giá trị w, b đã thử
plt.plot(w1, b1, 'ro', markersize=10, label=f'Mô hình 1: Cost={cost1:.2f}')
plt.plot(w2, b2, 'go', markersize=10, label=f'Mô hình 2: Cost={cost2:.2f}')
plt.plot(w3, b3, 'yo', markersize=10, label=f'Mô hình 3: Cost={cost3:.2f}')
plt.plot(w4, b4, 'mo', markersize=10, label=f'Mô hình 4: Cost={cost4:.2f}')

plt.grid(True)
plt.legend()
plt.show()

def plot_cost_3d(x_train, y_train, w_range=(100, 300), b_range=(-200, 200)):
    """
    Vẽ cost function J(w,b) trong không gian 3D
    
    Args:
        x_train: dữ liệu đầu vào
        y_train: dữ liệu đầu ra
        w_range: khoảng giá trị của w (hệ số góc)
        b_range: khoảng giá trị của b (hệ số tự do)
    """
    # Tạo lưới các giá trị w và b
    w_values = np.linspace(w_range[0], w_range[1], 50)
    b_values = np.linspace(b_range[0], b_range[1], 50)
    W, B = np.meshgrid(w_values, b_values)
    
    # Tính cost function cho mỗi cặp giá trị (w, b)
    Z = np.zeros_like(W)
    for i in range(len(w_values)):
        for j in range(len(b_values)):
            Z[j, i] = compute_cost(x_train, y_train, W[j, i], B[j, i])
    
    # Tìm giá trị tối ưu
    min_cost_idx = np.unravel_index(Z.argmin(), Z.shape)
    w_opt = W[min_cost_idx]
    b_opt = B[min_cost_idx]
    min_cost = Z[min_cost_idx]
    
    # Tạo đồ thị 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Vẽ bề mặt với gradient màu
    surface = ax.plot_surface(W, B, Z, cmap='viridis',
                            linewidth=0, antialiased=True)
    
    # Đánh dấu điểm tối ưu
    ax.scatter([w_opt], [b_opt], [min_cost],
              color='red', s=100, label='Điểm tối ưu')
    
    # Thêm các thành phần trang trí
    ax.set_xlabel('w (hệ số góc)')
    ax.set_ylabel('b (hệ số tự do)')
    ax.set_zlabel('J(w,b) (cost)')
    plt.colorbar(surface, label='Giá trị cost')
    plt.title('Cost Function J(w,b) trong Không Gian 3D')
    plt.legend()
    
    # In thông tin về điểm tối ưu
    print(f"\nTham số tối ưu tìm được:")
    print(f"  w* = {w_opt:.1f}")
    print(f"  b* = {b_opt:.1f}")
    print(f"  J(w*,b*) = {min_cost:.1f}")
    
    # Cho phép xoay đồ thị
    plt.show()