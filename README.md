# Ứng Dụng Giải Tích và Học Máy

Bộ ví dụ minh họa về các phép toán vector và ứng dụng trong học máy, tập trung vào cost function và gradient descent.

## Cấu Trúc Dự Án

```
calculus-machine-learning/
├── README.md                 # Tài liệu hướng dẫn và mô tả dự án
├── requirements.txt          # Danh sách các thư viện phụ thuộc
├── setup.py                  # File cấu hình cho việc cài đặt package
├── src/
│   └── calculus_ml/         # Package chính
│       ├── __init__.py      # Khởi tạo package và export các hàm chính
│       ├── core/            # Module chứa các hàm tính toán cốt lõi
│       │   ├── vector.py    # Các phép toán vector cơ bản (cộng, trừ, tích vô hướng)
│       │   ├── cost_function.py    # Tính toán cost function và tạo dữ liệu mẫu
│       │   └── gradient_descent.py # Cài đặt thuật toán gradient descent
│       └── visualization/    # Module chứa các hàm vẽ đồ thị
│           ├── cost_plot.py        # Vẽ đồ thị cost function và kết quả hồi quy
│           └── gradient_plot.py    # Vẽ đồ thị quá trình gradient descent
└── images/                  # Thư mục lưu các hình ảnh được tạo ra
    ├── cost_function_3d.png        # Bề mặt cost function trong không gian 3D
    ├── cost_function_contour.png   # Đường đồng mức của cost function
    ├── gradient_descent_3d.png     # Quá trình gradient descent trên bề mặt 3D
    ├── gradient_descent_contour.png # Quá trình gradient descent trên contour
    ├── gradient_descent_steps.png   # Các bước của gradient descent
    └── cost_history.png            # Lịch sử cost function qua các iteration
```

## Tính Năng Chính

1. **Phép Toán Vector Cơ Bản**
   - Cộng, trừ vector
   - Nhân vector với số vô hướng
   - Tích vô hướng (dot product)
   - Chuẩn hóa vector (normalization)

2. **Cost Function trong Học Máy**
   - Tính toán cost function J(w,b)
   - Trực quan hóa bề mặt cost 3D
   - Vẽ đường đồng mức của cost function
   - Hỗ trợ cả mô hình 1 tham số và 2 tham số

3. **Gradient Descent**
   - Tính gradient của cost function
   - Cài đặt thuật toán gradient descent
   - Trực quan hóa quá trình tối ưu
   - Theo dõi sự hội tụ qua các iteration

## Chi Tiết Các Ví Dụ

### 1. Ví dụ với 1 tham số
- **Dữ liệu**: Giá nhà dựa trên kích thước
- **Mô hình**: y = wx + b
- **Tham số**: w (giá/sqft), b (giá cơ bản)
- **Trực quan hóa**: 
  - Cost function 3D
  - Đường đồng mức
  - Quá trình gradient descent

### 2. Ví dụ với 2 tham số
- **Dữ liệu**: Giá nhà dựa trên kích thước và số phòng ngủ
- **Mô hình**: y = w₁x₁ + w₂x₂ + b
- **Tham số**: 
  - w₁ (giá/sqft)
  - w₂ (giá/phòng ngủ)
  - b (giá cơ bản)
- **Trực quan hóa**:
  - Cost function trong không gian nhiều chiều
  - Quá trình tối ưu với nhiều tham số

## Cài Đặt và Sử Dụng

1. Cài đặt:
```bash
pip install -r requirements.txt
pip install -e .
```

2. Chạy chương trình:
```bash
python -m calculus_ml.main
```

## Giấy Phép

MIT License