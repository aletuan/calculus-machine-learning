import os
from .cost_function import (
    generate_house_data,
    compute_cost,
    plot_house_data,
    plot_different_models,
    plot_model_errors,
    plot_squared_errors,
    plot_cost_contour
)

def main():
    # Tạo thư mục images nếu chưa tồn tại
    if not os.path.exists('images'):
        os.makedirs('images')
    
    # 1. Tạo dữ liệu
    x_train, y_train = generate_house_data()
    
    # 2. Vẽ dữ liệu ban đầu
    plot_house_data(x_train, y_train)
    
    # 3. Tính cost function cho các mô hình khác nhau
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
    
    # 4. Vẽ các mô hình khác nhau
    plot_different_models(x_train, y_train, models)
    
    # 5. Vẽ chi tiết sai số cho mô hình đầu tiên
    w, b, cost, _, _ = models[0]
    plot_model_errors(x_train, y_train, w, b, cost)
    
    # 6. Vẽ trực quan hóa bình phương sai số
    plot_squared_errors(x_train, y_train, w, b, cost)
    
    # 7. Vẽ đường đồng mức của Cost Function
    plot_cost_contour(x_train, y_train, models)
    
    print("\nHình ảnh kết quả đã được lưu trong thư mục 'images':")
    print("- house_data.png: Dữ liệu giá nhà")
    print("- different_models.png: Các mô hình dự đoán khác nhau")
    print("- model_errors.png: Chi tiết sai số của mô hình")
    print("- squared_errors.png: Trực quan hóa bình phương sai số")
    print("- cost_contour.png: Đường đồng mức của Cost Function")

if __name__ == "__main__":
    main() 