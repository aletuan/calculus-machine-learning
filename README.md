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

### 1. Linear Regression
- **Dữ liệu**: Giá nhà dựa trên kích thước
- **Mô hình**: y = wx + b
- **Tham số**: 
  - w: giá tăng theo mỗi 1000 sqft
  - b: giá cơ bản
- **Trực quan hóa**: 
  - Đường hồi quy tối ưu
  - Lịch sử cost function

### 2. Logistic Regression
- **Dữ liệu**: Kết quả tuyển sinh dựa trên điểm thi và GPA
- **Mô hình**: P(y=1) = g(w₁x₁ + w₂x₂ + b)
- **Tham số**: 
  - w₁: trọng số điểm thi
  - w₂: trọng số GPA
  - b: độ chệch
- **Trực quan hóa**:
  - Decision boundary
  - Lịch sử cost function

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