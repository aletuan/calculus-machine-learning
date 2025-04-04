# Regression Models in Machine Learning

Triển khai và trực quan hóa hai mô hình học máy cơ bản: Linear Regression và Logistic Regression, với các ví dụ thực tế về dự đoán giá nhà và phân loại kết quả tuyển sinh.

## Giới Thiệu Khái Niệm

### Machine Learning là gì?
Machine Learning (Học máy) là một nhánh của trí tuệ nhân tạo, cho phép máy tính "học" từ dữ liệu mà không cần được lập trình một cách tường minh. Trong project này, chúng ta tập trung vào học có giám sát (supervised learning) với hai dạng bài toán phổ biến:

### 1. Regression (Hồi quy)
- **Định nghĩa**: Dự đoán một giá trị liên tục dựa trên các đặc trưng đầu vào
- **Ví dụ**: Dự đoán giá nhà, dự báo nhiệt độ, dự đoán doanh thu
- **Đánh giá**: Sử dụng các metrics như MSE (Mean Squared Error), MAE (Mean Absolute Error)

### 2. Classification (Phân loại)
- **Định nghĩa**: Phân loại dữ liệu vào các nhóm/lớp rời rạc
- **Ví dụ**: Phân loại email (spam/không spam), chẩn đoán bệnh (có/không)
- **Đánh giá**: Accuracy, Precision, Recall, F1-score

### Các Khái Niệm Quan Trọng

1. **Mô hình (Model)**
   - Hàm số toán học mô tả mối quan hệ giữa đầu vào và đầu ra
   - Được xác định bởi các tham số (parameters) cần tối ưu

2. **Học (Learning)**
   - **Gradient Descent**: Thuật toán tối ưu tham số bằng cách di chuyển ngược hướng gradient
   - **Learning Rate**: Tốc độ học, quyết định độ lớn của bước cập nhật trong gradient descent
   - **Iteration**: Số lần lặp lại quá trình học

3. **Đánh giá (Evaluation)**
   - **Loss Function**: Hàm đo lường độ sai lệch giữa dự đoán và giá trị thực
   - **Training**: Quá trình tối ưu mô hình trên tập dữ liệu huấn luyện
   - **Convergence**: Sự hội tụ của quá trình học, thể hiện qua đồ thị loss function

### Các Công Cụ Sử Dụng
- **NumPy**: Thư viện tính toán số học
- **Matplotlib**: Thư viện vẽ đồ thị
- **Rich**: Thư viện định dạng output trong terminal

## Cấu Trúc Dự Án

```
calculus-machine-learning/
├── README.md                 # Tài liệu hướng dẫn và mô tả dự án
├── requirements.txt          # Danh sách các thư viện phụ thuộc
├── setup.py                  # File cấu hình cho việc cài đặt package
├── src/
│   └── calculus_ml/         # Package chính
│       ├── __init__.py      # Khởi tạo package
│       ├── core/            # Module chứa các hàm tính toán cốt lõi
│       │   ├── base/        # Các class và hàm cơ sở
│       │   ├── linear/      # Linear regression
│       │   └── logistic/    # Logistic regression
│       ├── visualization/    # Module chứa các hàm vẽ đồ thị
│       │   ├── base/        # Các hàm vẽ cơ bản
│       │   ├── linear/      # Vẽ cho linear regression
│       │   └── logistic/    # Vẽ cho logistic regression
│       └── examples/        # Các ví dụ minh họa
└── images/                  # Thư mục lưu các hình ảnh được tạo ra
    ├── linear_regression_fit.png    # Đường hồi quy tuyến tính và dữ liệu
    ├── linear_cost_history.png      # Lịch sử cost function của linear regression
    ├── logistic_decision_boundary.png # Decision boundary của logistic regression
    └── logistic_cost_history.png     # Lịch sử cost function của logistic regression
```

## Ví Dụ Minh Họa

### 1. Linear Regression - Dự Đoán Giá Nhà

#### Mô tả bài toán
- **Mục tiêu**: Dự đoán giá nhà dựa trên diện tích
- **Dữ liệu**: 
  - 100 mẫu nhà với diện tích và giá
  - Biến đầu vào (x): Diện tích (1000 sqft), phân phối chuẩn quanh 2.5 (1-5)
  - Biến đầu ra (y): Giá nhà (1000$), tương quan tuyến tính với diện tích

#### Công thức toán học
- **Mô hình dự đoán**: y = wx + b
  - w: hệ số ảnh hưởng của diện tích đến giá
  - b: giá cơ bản của nhà
- **Hàm mất mát**: J(w,b) = (1/2m) * Σ(f(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
- **Gradient descent**: 
  - ∂J/∂w = (1/m) * Σ(f(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x⁽ⁱ⁾
  - ∂J/∂b = (1/m) * Σ(f(x⁽ⁱ⁾) - y⁽ⁱ⁾)

#### Trực quan hóa kết quả
![Linear Regression Fit](images/linear_regression_fit.png)
- **Đồ thị dự đoán**:
  - Điểm xanh: dữ liệu thực tế (diện tích, giá)
  - Đường đỏ: mô hình dự đoán tối ưu
  - Trục x: Diện tích nhà (1000 sqft)
  - Trục y: Giá nhà (1000$)

![Linear Cost History](images/linear_cost_history.png)
- **Đồ thị huấn luyện**:
  - Trục x: Số vòng lặp
  - Trục y: Giá trị hàm mất mát
  - Đường cong giảm thể hiện mô hình đang học tốt

### 2. Logistic Regression - Dự Đoán Kết Quả Tuyển Sinh

#### Mô tả bài toán
- **Mục tiêu**: Dự đoán kết quả tuyển sinh (Đỗ/Trượt) dựa trên điểm thi và GPA
- **Dữ liệu**:
  - 100 hồ sơ tuyển sinh
  - Biến đầu vào:
    - x₁: Điểm thi (0-100), phân phối chuẩn quanh 65
    - x₂: GPA (0-4), phân phối chuẩn quanh 3.0
  - Biến đầu ra: Kết quả (1: Đỗ, 0: Trượt)

#### Công thức toán học
- **Hàm sigmoid**: g(z) = 1 / (1 + e^(-z))
- **Mô hình dự đoán**: P(đỗ) = g(w₁x₁ + w₂x₂ + b)
  - w₁: trọng số của điểm thi
  - w₂: trọng số của GPA
  - b: độ chệch
- **Binary cross-entropy loss**: 
  J(w₁,w₂,b) = -(1/m) * Σ[y⁽ⁱ⁾log(h(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h(x⁽ⁱ⁾))]
- **Gradient descent**:
  - ∂J/∂w₁ = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x₁⁽ⁱ⁾
  - ∂J/∂w₂ = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x₂⁽ⁱ⁾
  - ∂J/∂b = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)

#### Trực quan hóa kết quả
![Decision Boundary](images/logistic_decision_boundary.png)
- **Đồ thị phân loại**:
  - Điểm xanh dương: Sinh viên đỗ
  - Điểm đỏ: Sinh viên trượt
  - Đường xanh lá: Decision boundary (P(đỗ) = 0.5)
  - Vùng màu: Xác suất đỗ (từ 0 đến 1)
  - Trục x: Điểm thi
  - Trục y: GPA

![Logistic Cost History](images/logistic_cost_history.png)
- **Đồ thị huấn luyện**:
  - Trục x: Số vòng lặp
  - Trục y: Giá trị hàm mất mát
  - Đường cong giảm thể hiện mô hình đang học tốt

## Chi Tiết Triển Khai

### Các Module Chính

1. **Core Module**: Cài đặt các thuật toán
```python
class LinearRegression:
    """Mô hình hồi quy tuyến tính: y = wx + b"""
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def compute_cost(self, X, y):
        m = len(y)
        predictions = self.predict(X)
        return (1/(2*m)) * np.sum((predictions - y)**2)

class LogisticRegression:
    """Mô hình phân loại nhị phân: P(y=1) = g(w₁x₁ + w₂x₂ + b)"""
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, X, y):
        m = len(y)
        predictions = self.predict(X)
        return -(1/m) * np.sum(y*np.log(predictions) + (1-y)*np.log(1-predictions))
```

2. **Visualization Module**: Trực quan hóa kết quả
- Vẽ dữ liệu và mô hình dự đoán
- Vẽ đồ thị hội tụ của hàm mất mát
- Tạo các đồ thị tương tác

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