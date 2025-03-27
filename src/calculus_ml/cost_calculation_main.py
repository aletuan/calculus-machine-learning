import os
import numpy as np
import matplotlib.pyplot as plt
from .cost_function import (
    generate_house_data,
    compute_cost,
    plot_house_data,
    plot_different_models,
    plot_model_errors,
    plot_squared_errors,
    plot_cost_contour
)
from .cost_calculation import predict, plot_cost_3d

def main():
    # Tạo thư mục images nếu chưa tồn tại
    if not os.path.exists('images'):
        os.makedirs('images')
    
    # Tạo dữ liệu mẫu
    x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # kích thước (1000 sqft)
    y_train = np.array([300, 500, 700, 900, 1100])  # giá (1000s $)

    # 1. Hiển thị dữ liệu
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, color='blue', s=100, label='Dữ liệu nhà')
    plt.xlabel('Kích thước (1000 sqft)')
    plt.ylabel('Giá (1000s $)')
    plt.title('Dữ liệu giá nhà')
    plt.grid(True)
    plt.legend()
    plt.show()

    # 2. Thử nghiệm với các giá trị w, b khác nhau
    test_params = [
        (100, 100),  # w=100, b=100
        (200, 100),  # w=200, b=100
        (200, 0),    # w=200, b=0
        (200, -100)  # w=200, b=-100
    ]

    print("\nKết quả với các tham số khác nhau:")
    for w, b in test_params:
        cost = compute_cost(x_train, y_train, w, b)
        print(f"w = {w}, b = {b} => Cost = {cost:.2f}")

    # 3. Vẽ đồ thị 3D của cost function
    print("\nVẽ cost function trong không gian 3D...")
    plot_cost_3d(x_train, y_train, w_range=(100, 300), b_range=(-200, 200))

    # 4. Tính cost function cho các mô hình khác nhau
    print("Tính cost function với các giá trị w, b khác nhau:")
    
    # Định nghĩa các mô hình cần thử nghiệm
    models = [
        (100, 100, 0, 'r-', 'Mô hình 1'),
        (200, 100, 0, 'g-', 'Mô hình 2'),
        (200, 0, 0, 'y-', 'Mô hình 3'),
        (200, -100, 0, 'm-', 'Mô hình 4')
    ]
    
    # Tính cost cho từng mô hình
    for i, (w, b, _, color, label) in enumerate(models):
        cost = compute_cost(x_train, y_train, w, b)
        models[i] = (w, b, cost, color, label)
        print(f"w = {w}, b = {b} => Cost = {cost:.2f}")
    
    # 5. Vẽ các mô hình khác nhau
    plot_different_models(x_train, y_train, models)
    
    # 6. Vẽ chi tiết sai số cho mô hình đầu tiên
    w, b, cost, _, _ = models[0]
    plot_model_errors(x_train, y_train, w, b, cost)
    
    # 7. Vẽ trực quan hóa bình phương sai số
    plot_squared_errors(x_train, y_train, w, b, cost)
    
    # 8. Vẽ đường đồng mức của Cost Function
    plot_cost_contour(x_train, y_train, models)
    
    print("\nHình ảnh kết quả đã được lưu trong thư mục 'images':")
    print("- house_data.png: Dữ liệu giá nhà")
    print("- different_models.png: Các mô hình dự đoán khác nhau")
    print("- model_errors.png: Chi tiết sai số của mô hình")
    print("- squared_errors.png: Trực quan hóa bình phương sai số")
    print("- cost_contour.png: Đường đồng mức của Cost Function")

if __name__ == "__main__":
    main() 