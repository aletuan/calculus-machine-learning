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

2. **Đánh giá (Evaluation)**
   - **Loss Function (Hàm mất mát)**: 
     - **Định nghĩa**: Hàm số đo lường mức độ sai lệch giữa giá trị dự đoán của mô hình và giá trị thực tế
     - **Vai trò**:
       - Đóng vai trò là "la bàn" định hướng quá trình học của mô hình
       - Cung cấp một giá trị số (scalar) có thể tối ưu hóa bằng gradient descent
       - Là cầu nối giữa lý thuyết toán học và ứng dụng thực tế của mô hình
     
     - **Các loại hàm mất mát phổ biến**:
       1. **Mean Squared Error (MSE)**:
          - Công thức: MSE = (1/m) * Σ(y_pred - y_true)²
          - Phù hợp với: Linear Regression và các bài toán dự đoán giá trị liên tục
          - Đặc điểm: Phạt nặng các sai số lớn (do bình phương), nhạy cảm với outliers
       
       2. **Binary Cross-Entropy**:
          - Công thức: BCE = -(1/m) * Σ[y_true*log(y_pred) + (1-y_true)*log(1-y_pred)]
          - Phù hợp với: Logistic Regression và các bài toán phân loại nhị phân
          - Đặc điểm: Đánh giá tốt xác suất dự đoán, phù hợp với dữ liệu nhãn 0/1
       
       3. **Categorical Cross-Entropy**:
          - Phù hợp với: Bài toán phân loại nhiều lớp
          - Mở rộng của Binary Cross-Entropy cho nhiều lớp
       
       4. **Hinge Loss**:
          - Sử dụng cho: Support Vector Machines
          - Tối ưu hóa biên phân loại (margin)
     
     - **Lựa chọn hàm mất mát**:
       - Phụ thuộc vào bản chất của bài toán (hồi quy/phân loại)
       - Phụ thuộc vào phân phối của dữ liệu
       - Ảnh hưởng trực tiếp đến quá trình học và kết quả cuối cùng của mô hình
   
   - **Training**: Quá trình tối ưu mô hình trên tập dữ liệu huấn luyện
   - **Convergence**: Sự hội tụ của quá trình học, thể hiện qua đồ thị loss function

3. **Học (Learning)**
   - **Gradient Descent**: Thuật toán tối ưu tham số quan trọng nhất trong học máy
     - **Gradient là gì?**
       - Gradient là vector chứa các đạo hàm riêng của hàm mất mát theo từng tham số
       - Gradient chỉ ra hướng tăng nhanh nhất của hàm mất mát tại một điểm
       - Ví dụ: Với hàm mất mát J(w,b), gradient là [∂J/∂w, ∂J/∂b]
     
     - **Tại sao đi ngược hướng gradient?**
       - Mục tiêu: Tìm tham số để hàm mất mát đạt giá trị nhỏ nhất
       - Gradient chỉ hướng tăng nhanh nhất → Ngược gradient là hướng giảm nhanh nhất
       - Giống như đi xuống đồi: luôn chọn hướng dốc nhất để xuống nhanh nhất
     
     - **Quá trình cập nhật tham số**:
       1. Tính gradient tại vị trí hiện tại
       2. Cập nhật tham số: θ_new = θ_old - α * gradient
          - α (learning rate): điều chỉnh độ lớn của bước di chuyển
          - Nếu α quá lớn: có thể vượt qua điểm tối ưu
          - Nếu α quá nhỏ: hội tụ chậm
     
     - **Ví dụ trực quan**:
       ![Gradient Descent Example](images/gradient_descent_example.png)
       - **Đồ thị minh họa**:
         - Đường cong xanh: Hàm mất mát (Loss function)
         - Điểm đỏ: Vị trí hiện tại của tham số
         - Đường đứt nét đỏ: Gradient (độ dốc) tại điểm hiện tại
         - Mũi tên xanh lá: Hướng cập nhật tham số (ngược với gradient)
       
       - **Giải thích**:
         - Gradient dương (độ dốc đi lên) → Di chuyển sang trái (giảm tham số)
         - Gradient âm (độ dốc đi xuống) → Di chuyển sang phải (tăng tham số)
         - Quá trình lặp lại cho đến khi đạt điểm có gradient ≈ 0 (điểm tối ưu)

   - **Learning Rate**: Tốc độ học, quyết định độ lớn của bước cập nhật trong gradient descent
   - **Iteration**: Số lần lặp lại quá trình học

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
├── .gitignore               # Các file và thư mục cần bỏ qua khi commit
├── src/
│   └── calculus_ml/         # Package chính
│       ├── __init__.py      # Khởi tạo package
│       ├── main.py          # File chính chứa các ví dụ và chạy thử nghiệm
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
    ├── gradient_descent_example.png    # Minh họa quá trình gradient descent
    ├── linear_regression_fit.png       # Đường hồi quy tuyến tính và dữ liệu
    ├── linear_cost_history.png         # Lịch sử cost function của linear regression
    ├── multiple_regression_fit.png     # Mặt phẳng hồi quy nhiều biến và dữ liệu
    ├── multiple_cost_history.png       # Lịch sử cost function của hồi quy nhiều biến
    ├── logistic_decision_boundary.png  # Decision boundary của logistic regression
    └── logistic_cost_history.png       # Lịch sử cost function của logistic regression
```

## Ví Dụ Minh Họa

### 1. Linear Regression (Hồi quy tuyến tính)

#### 1.1. Hồi quy tuyến tính đơn giản - Dự Đoán Giá Nhà

##### Mô tả bài toán
- **Mục tiêu**: Dự đoán giá nhà dựa trên diện tích
- **Dữ liệu**: 
  - 100 mẫu nhà với diện tích và giá
  - Biến đầu vào (x): Diện tích (1000 sqft), phân phối chuẩn quanh 2.5 (1-5)
  - Biến đầu ra (y): Giá nhà (1000$), tương quan tuyến tính với diện tích

##### Công thức toán học
- **Mô hình dự đoán**: y = wx + b
  - w: hệ số ảnh hưởng của diện tích đến giá
  - b: giá cơ bản của nhà
- **Hàm mất mát**: J(w,b) = (1/2m) * Σ(f(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
- **Gradient descent**: 
  - ∂J/∂w = (1/m) * Σ(f(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x⁽ⁱ⁾
  - ∂J/∂b = (1/m) * Σ(f(x⁽ⁱ⁾) - y⁽ⁱ⁾)

##### Trực quan hóa kết quả
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

#### 1.2. Hồi quy tuyến tính nhiều biến - Dự Đoán Giá Nhà

##### Mô tả bài toán
- **Mục tiêu**: Dự đoán giá nhà dựa trên diện tích và số phòng ngủ
- **Dữ liệu**: 
  - 100 mẫu nhà với diện tích, số phòng ngủ và giá
  - Biến đầu vào:
    - x₁: Diện tích (1000 sqft)
    - x₂: Số phòng ngủ (1-5)
  - Biến đầu ra (y): Giá nhà (1000$)

##### Công thức toán học
- **Mô hình dự đoán**: y = w₁x₁ + w₂x₂ + b
  - w₁: hệ số ảnh hưởng của diện tích đến giá
  - w₂: hệ số ảnh hưởng của số phòng ngủ đến giá
  - b: giá cơ bản của nhà
- **Hàm mất mát**: J(w₁,w₂,b) = (1/2m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
- **Gradient descent**: 
  - ∂J/∂w₁ = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x₁⁽ⁱ⁾
  - ∂J/∂w₂ = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x₂⁽ⁱ⁾
  - ∂J/∂b = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)

##### Trực quan hóa kết quả
![Multiple Regression Fit](images/multiple_regression_fit.png)
- **Đồ thị dự đoán**:
  - Điểm xanh: dữ liệu thực tế (diện tích, số phòng ngủ, giá)
  - Mặt phẳng đỏ: mô hình dự đoán tối ưu
  - Trục x: Diện tích nhà (1000 sqft)
  - Trục y: Số phòng ngủ
  - Trục z: Giá nhà (1000$)

![Multiple Cost History](images/multiple_cost_history.png)
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