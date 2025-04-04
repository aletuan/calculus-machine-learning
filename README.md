# Ứng Dụng Giải Tích và Học Máy

Tập hợp ví dụ minh họa về các thuật toán học máy cơ bản, tập trung vào linear regression và logistic regression.

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

## Tính Năng Chính

1. **Linear Regression**
   - Mô hình hồi quy tuyến tính một biến
   - Tối ưu hóa bằng gradient descent
   - Trực quan hóa đường hồi quy và quá trình học

2. **Logistic Regression**
   - Mô hình phân loại nhị phân
   - Tối ưu hóa bằng gradient descent
   - Trực quan hóa decision boundary và quá trình học

## Chi Tiết Các Module

### 1. Linear Regression

```python
class LinearRegression:
    """
    Mô hình hồi quy tuyến tính
    y = wx + b
    
    Thuộc tính:
        weights: Trọng số w
        bias: Độ chệch b
        learning_rate: Tốc độ học α
        num_iterations: Số vòng lặp
    """
    
    def predict(self, X):
        """Dự đoán giá trị y = w*X + b"""
        return np.dot(X, self.weights) + self.bias
    
    def compute_cost(self, X, y):
        """Tính MSE cost function"""
        m = len(y)
        predictions = self.predict(X)
        return (1/(2*m)) * np.sum((predictions - y)**2)
```

### 2. Logistic Regression

```python
class LogisticRegression:
    """
    Mô hình phân loại nhị phân
    P(y=1) = g(w₁x₁ + w₂x₂ + b)
    với g(z) = 1/(1+e^(-z))
    
    Thuộc tính:
        weights: Trọng số [w₁, w₂]
        bias: Độ chệch b
        learning_rate: Tốc độ học α
        num_iterations: Số vòng lặp
    """
    
    def predict(self, X):
        """Dự đoán xác suất P(y=1)"""
        z = np.dot(X, self.weights) + self.bias
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, X, y):
        """Binary cross-entropy loss"""
        m = len(y)
        predictions = self.predict(X)
        return -(1/m) * np.sum(y*np.log(predictions) + (1-y)*np.log(1-predictions))
```

## Minh Họa Trực Quan

1. **Linear Regression**
![Linear Regression Fit](images/linear_regression_fit.png)
- Dữ liệu: Điểm và đường hồi quy tối ưu
- Đường màu đỏ: mô hình dự đoán

![Linear Cost History](images/linear_cost_history.png)
- Sự hội tụ của cost function
- Ảnh hưởng của learning rate

2. **Logistic Regression**
![Decision Boundary](images/logistic_decision_boundary.png)
- Dữ liệu phân loại (đỗ/trượt)
- Đường màu xanh: decision boundary

![Logistic Cost History](images/logistic_cost_history.png)
- Sự hội tụ của binary cross-entropy loss
- Quá trình tối ưu tham số

## Ví dụ Minh Họa

### 1. Linear Regression - Dự Đoán Giá Nhà
- **Dữ liệu**: 
  - Tập dữ liệu mô phỏng về giá nhà (100 mẫu)
  - Biến đầu vào: Kích thước nhà (1000 sqft), phân phối chuẩn quanh 2.5 (1-5)
  - Biến đầu ra: Giá nhà (1000$), tương quan tuyến tính với kích thước
- **Mô hình**: 
  - Phương trình: y = wx + b
  - Hàm mất mát: J(w,b) = (1/2m) * Σ(f(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
  - Gradient descent: 
    - ∂J/∂w = (1/m) * Σ(f(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x⁽ⁱ⁾
    - ∂J/∂b = (1/m) * Σ(f(x⁽ⁱ⁾) - y⁽ⁱ⁾)
- **Trực quan hóa**: 
  - Scatter plot dữ liệu thực tế
  - Đường hồi quy tối ưu (đỏ)
  - Đồ thị hội tụ của hàm mất mát

### 2. Logistic Regression - Dự Đoán Kết Quả Tuyển Sinh
- **Dữ liệu**:
  - Tập dữ liệu mô phỏng về tuyển sinh (100 mẫu)
  - Biến đầu vào:
    - x₁: Điểm thi (0-100), phân phối chuẩn quanh 65
    - x₂: GPA (0-4), phân phối chuẩn quanh 3.0
  - Biến đầu ra: Kết quả (Đỗ/Trượt)
- **Mô hình**:
  - Hàm sigmoid: g(z) = 1 / (1 + e^(-z))
  - Hàm dự đoán: P(đỗ) = g(w₁x₁ + w₂x₂ + b)
  - Binary cross-entropy loss: J(w₁,w₂,b) = -(1/m) * Σ[y⁽ⁱ⁾log(h(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h(x⁽ⁱ⁾))]
  - Gradient descent:
    - ∂J/∂w₁ = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x₁⁽ⁱ⁾
    - ∂J/∂w₂ = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x₂⁽ⁱ⁾
    - ∂J/∂b = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)
- **Trực quan hóa**:
  - Scatter plot phân loại sinh viên (Đỗ: xanh dương, Trượt: đỏ)
  - Decision boundary (đường xanh lá)
  - Contour plot thể hiện xác suất đỗ
  - Đồ thị hội tụ của hàm mất mát

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