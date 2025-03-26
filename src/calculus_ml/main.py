import os
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

def main():
    # Tạo thư mục images nếu chưa tồn tại
    if not os.path.exists('images'):
        os.makedirs('images')
    
    # 1. Vector cơ bản và biểu diễn trực quan
    v1, v2 = create_vectors()
    vec_ops = vector_operations(v1, v2)
    plot_basic_vectors(v1, v2, vec_ops['v_sum'], vec_ops['v_diff'])
    
    # In kết quả phép toán vector
    print("===== PHÉP TOÁN VECTOR CƠ BẢN =====")
    print(f"Vector v1: {v1}")
    print(f"Vector v2: {v2}")
    print(f"\nCộng vector (v1 + v2): {vec_ops['v_sum']}")
    print(f"Trừ vector (v1 - v2): {vec_ops['v_diff']}")
    print(f"Nhân vector với scalar ({vec_ops['scalar']} * v1): {vec_ops['v_scaled']}")
    print(f"Độ dài của v1: {vec_ops['v1_norm']}")
    print(f"Tích vô hướng (v1 · v2): {vec_ops['dot_product']}")
    print(f"Tích có hướng (v1 × v2): {vec_ops['cross_product']}")
    
    # 2. Vector đơn vị và góc giữa hai vector
    vec_unit_angle = unit_vector_and_angle(v1, v2)
    print(f"\nVector đơn vị của v1: {vec_unit_angle['v1_unit']}")
    print(f"Độ dài của vector đơn vị: {vec_unit_angle['v1_unit_norm']}")
    print(f"Góc giữa v1 và v2: {vec_unit_angle['angle']:.2f} độ")
    
    plot_vector_angle(v1, v2, vec_unit_angle['angle'], vec_unit_angle['cos_theta'])
    
    # 3. Ứng dụng trong machine learning: Linear Regression
    x, y = generate_data()
    w, b = fit_linear_regression(x, y)
    
    print("\n===== ỨNG DỤNG TRONG LINEAR REGRESSION =====")
    print(f"Tham số tìm được: w = {w:.4f}, b = {b:.4f}")
    print(f"Phương trình hồi quy: y = {w:.4f}x + {b:.4f}")
    
    plot_linear_regression(x, y, w, b)
    
    # 4. Chiếu vector và ứng dụng
    vec_proj = vector_projection(v1, v2)
    print("\n===== CHIẾU VECTOR =====")
    print(f"Chiếu v1 lên v2: {vec_proj['projection_vector']}")
    print(f"Độ dài của vector chiếu: {vec_proj['projection_norm']}")
    
    plot_vector_projection(v1, v2, vec_proj['projection_vector'])
    
    print("\nHình ảnh kết quả đã được lưu trong thư mục 'images':")

if __name__ == "__main__":
    main() 