import numpy as np
import matplotlib.pyplot as plt

# Thiết lập vẽ đẹp hơn
plt.style.use('seaborn-v0_8')
plt.figure(figsize=(10, 6))

###########################################
# 1. Vector cơ bản và biểu diễn trực quan
###########################################

# Tạo một số vector 2D đơn giản
v1 = np.array([3, 4])  # Vector [3, 4]
v2 = np.array([2, 1])  # Vector [2, 1]

# Thiết lập không gian vẽ
plt.figure(figsize=(10, 6))
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(alpha=0.3)

# Vẽ các vector bắt đầu từ gốc tọa độ
plt.arrow(0, 0, v1[0], v1[1], head_width=0.2, head_length=0.3, fc='blue', ec='blue', label='v1 = [3, 4]')
plt.arrow(0, 0, v2[0], v2[1], head_width=0.2, head_length=0.3, fc='red', ec='red', label='v2 = [2, 1]')

# Tính và vẽ tổng vector
v_sum = v1 + v2
plt.arrow(0, 0, v_sum[0], v_sum[1], head_width=0.2, head_length=0.3, fc='green', ec='green', label='v1 + v2 = [5, 5]')

# Tính và vẽ hiệu vector
v_diff = v1 - v2
plt.arrow(0, 0, v_diff[0], v_diff[1], head_width=0.2, head_length=0.3, fc='purple', ec='purple', label='v1 - v2 = [1, 3]')

# Hiện thị hình ảnh
plt.xlim(-1, 6)
plt.ylim(-1, 6)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Biểu diễn vector trong không gian 2D')
plt.legend()
plt.savefig('vectors_basic.png')
plt.close()

###########################################
# 2. Phép toán vector cơ bản
###########################################

print("===== PHÉP TOÁN VECTOR CƠ BẢN =====")
print(f"Vector v1: {v1}")
print(f"Vector v2: {v2}")

# Cộng vector
print(f"\nCộng vector (v1 + v2): {v1 + v2}")

# Trừ vector
print(f"Trừ vector (v1 - v2): {v1 - v2}")

# Nhân vector với một số
scalar = 2
print(f"Nhân vector với scalar ({scalar} * v1): {scalar * v1}")

# Độ dài (norm) của vector
v1_norm = np.linalg.norm(v1)
print(f"Độ dài của v1: {v1_norm}")

# Tích vô hướng (dot product)
dot_product = np.dot(v1, v2)
print(f"Tích vô hướng (v1 · v2): {dot_product}")

# Tích có hướng (cross product) - chỉ trong không gian 3D
v1_3d = np.array([v1[0], v1[1], 0])  # Thêm chiều z = 0
v2_3d = np.array([v2[0], v2[1], 0])  # Thêm chiều z = 0
cross_product = np.cross(v1_3d, v2_3d)
print(f"Tích có hướng (v1 × v2): {cross_product}")

###########################################
# 3. Vector đơn vị và góc giữa hai vector
###########################################

# Vector đơn vị của v1
v1_unit = v1 / np.linalg.norm(v1)
print(f"\nVector đơn vị của v1: {v1_unit}")
print(f"Độ dài của vector đơn vị: {np.linalg.norm(v1_unit)}")

# Góc giữa hai vector (công thức: cos(θ) = (v1·v2)/(|v1|*|v2|))
cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
angle = np.arccos(cos_theta) * 180 / np.pi  # Chuyển từ radian sang độ
print(f"Góc giữa v1 và v2: {angle:.2f} độ")

# Vẽ minh họa góc giữa hai vector
plt.figure(figsize=(8, 6))
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(alpha=0.3)

plt.arrow(0, 0, v1[0], v1[1], head_width=0.2, head_length=0.3, fc='blue', ec='blue', label='v1 = [3, 4]')
plt.arrow(0, 0, v2[0], v2[1], head_width=0.2, head_length=0.3, fc='red', ec='red', label='v2 = [2, 1]')

# Vẽ cung tròn để hiển thị góc
radius = 1
theta = np.linspace(0, np.arccos(cos_theta), 100)
x = radius * np.cos(theta)
y = radius * np.sin(theta)
plt.plot(x, y, 'g-', alpha=0.7)
plt.annotate(f'{angle:.1f}°', xy=(radius/2, radius/4), fontsize=12)

plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Góc giữa hai vector')
plt.legend()
plt.savefig('vector_angle.png')
plt.close()

###########################################
# 4. Ứng dụng trong machine learning: Linear Regression
###########################################

# Tạo dữ liệu đơn giản
np.random.seed(42)
x = np.array([1, 2, 3, 4, 5])
# y = 2x + 1 + noise
y = 2 * x + 1 + np.random.randn(5) * 0.5

# Trong linear regression đơn giản: y = wx + b
# Vector x và y có thể được coi là hai vector trong không gian dữ liệu

# Chuyển x thành ma trận thiết kế (thêm cột 1 cho bias)
X = np.column_stack((np.ones_like(x), x))

# Giải phương trình tuyến tính để tìm w và b
# (X^T * X)^(-1) * X^T * y
params = np.linalg.inv(X.T @ X) @ X.T @ y

# Lấy ra hệ số
b = params[0]  # bias
w = params[1]  # weight

print("\n===== ỨNG DỤNG TRONG LINEAR REGRESSION =====")
print(f"Tham số tìm được: w = {w:.4f}, b = {b:.4f}")
print(f"Phương trình hồi quy: y = {w:.4f}x + {b:.4f}")

# Vẽ dữ liệu và đường hồi quy
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Dữ liệu')
plt.plot(x, w*x + b, 'r-', label=f'y = {w:.4f}x + {b:.4f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.savefig('linear_regression.png')
plt.close()

###########################################
# 5. Chiếu vector và ứng dụng
###########################################

# Chiếu vector v1 lên v2
# proj_v1_onto_v2 = (v1·v2 / |v2|^2) * v2
projection_scalar = np.dot(v1, v2) / np.dot(v2, v2)
projection_vector = projection_scalar * v2

print("\n===== CHIẾU VECTOR =====")
print(f"Chiếu v1 lên v2: {projection_vector}")
print(f"Độ dài của vector chiếu: {np.linalg.norm(projection_vector)}")

# Vẽ minh họa phép chiếu
plt.figure(figsize=(10, 6))
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(alpha=0.3)

# Vẽ các vector gốc
plt.arrow(0, 0, v1[0], v1[1], head_width=0.2, head_length=0.3, fc='blue', ec='blue', label='v1 = [3, 4]')
plt.arrow(0, 0, v2[0], v2[1], head_width=0.2, head_length=0.3, fc='red', ec='red', label='v2 = [2, 1]')

# Vẽ vector chiếu
plt.arrow(0, 0, projection_vector[0], projection_vector[1], head_width=0.2, head_length=0.3, 
         fc='green', ec='green', label='proj_v1_onto_v2')

# Vẽ đường từ v1 đến vector chiếu
plt.plot([v1[0], projection_vector[0]], [v1[1], projection_vector[1]], 'k--', alpha=0.7)

plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Chiếu vector v1 lên v2')
plt.legend()
plt.savefig('vector_projection.png')
plt.close()

print("\nHình ảnh kết quả đã được lưu: 'vectors_basic.png', 'vector_angle.png', 'linear_regression.png', 'vector_projection.png'")