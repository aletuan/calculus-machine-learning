# Ứng Dụng Giải Tích và Học Máy

Bộ ví dụ minh họa về các phép toán vector và ứng dụng trong học máy.

## Cấu Trúc Dự Án

```
calculus-machine-learning/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   └── calculus_ml/
│       ├── __init__.py
│       ├── vector_operations.py
│       ├── vector_visualization.py
│       ├── linear_regression.py
│       ├── cost_calculation.py
│       ├── cost_calculation_main.py
│       └── cost_function.py
└── images/
    └── ...
```

## Tính Năng

1. **Vector cơ bản và biểu diễn trực quan**
   - Tạo và vẽ vector trong không gian 2D
   - Minh họa phép cộng và trừ vector

2. **Phép toán vector cơ bản**
   - Cộng, trừ vector
   - Nhân vector với số vô hướng
   - Tính độ dài (norm) của vector
   - Tích vô hướng (dot product)
   - Tích có hướng (cross product)

3. **Vector đơn vị và góc giữa hai vector**
   - Chuẩn hóa vector thành vector đơn vị
   - Tính góc giữa hai vector
   - Biểu diễn trực quan góc giữa vector

4. **Ứng dụng trong Linear Regression**
   - Tạo dữ liệu đơn giản
   - Sử dụng phép tính vector để tìm hệ số hồi quy
   - Vẽ đường hồi quy

5. **Chiếu vector và ứng dụng**
   - Chiếu vector lên một vector khác
   - Biểu diễn trực quan phép chiếu

6. **Tính toán Cost Function trong Học Máy**
   - Minh họa chi tiết cách tính cost function
   - So sánh các mô hình khác nhau
   - Trực quan hóa sai số và bình phương sai số
   - Vẽ đường đồng mức của cost function

## Chi Tiết Các Hàm Chính

### 1. Vector Operations (`vector_operations.py`)

```python
def add_vectors(v1, v2):
    """
    Cộng hai vector
    Ví dụ: add_vectors([1, 2], [3, 4]) -> [4, 6]
    """
    return [x + y for x, y in zip(v1, v2)]

def dot_product(v1, v2):
    """
    Tính tích vô hướng của hai vector
    Ví dụ: dot_product([1, 2], [3, 4]) -> 11
    """
    return sum(x * y for x, y in zip(v1, v2))
```

### 2. Cost Calculation (`cost_calculation.py`)

```python
def predict(x, w, b):
    """
    Dự đoán giá trị y dựa trên x, w, b
    y = w*x + b
    
    Ví dụ:
    x = 2.0, w = 200, b = 100
    predict(2.0, 200, 100) -> 500
    """
    return w * x + b

def compute_cost(x, y, w, b):
    """
    Tính cost function J(w,b)
    J(w,b) = (1/2m) * Σ(f(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
    
    Ví dụ:
    x = [1.0, 2.0], y = [300, 500], w = 200, b = 100
    compute_cost(x, y, w, b) -> 2500
    """
    m = len(x)
    total_cost = sum((predict(x[i], w, b) - y[i]) ** 2 for i in range(m))
    return total_cost / (2 * m)
```

## Ví Dụ Sử Dụng

### 1. Minh Họa Cost Function

```python
# Tạo dữ liệu mẫu
x_train = [1.0, 2.0, 3.0, 4.0, 5.0]  # kích thước nhà (1000 sqft)
y_train = [300, 500, 700, 900, 1100]  # giá nhà (1000s $)

# Tính cost cho các mô hình khác nhau
w1, b1 = 100, 100
cost1 = compute_cost(x_train, y_train, w1, b1)
print(f"Cost với w={w1}, b={b1}: {cost1:.2f}")

# Vẽ biểu đồ so sánh các mô hình
plot_different_models(x_train, y_train, [(w1, b1), (w2, b2), (w3, b3)])
```

### 2. Trực Quan Hóa Sai Số

```python
# Vẽ chi tiết sai số của mô hình
plot_model_errors(x_train, y_train, w, b)

# Vẽ bình phương sai số
plot_squared_errors(x_train, y_train, w, b)
```

## Hướng Dẫn Cài Đặt

1. Tải mã nguồn:
```bash
git clone https://github.com/aletuan/calculus-machine-learning.git
cd calculus-machine-learning
```

2. Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

3. Cài đặt gói phần mềm ở chế độ phát triển:
```bash
pip install -e .
```

## Cách Sử Dụng

Chạy chương trình chính để xem tất cả các ví dụ:
```bash
python -m calculus_ml.main
```

## Phát Triển

Để đóng góp cho dự án:

1. Fork repository này
2. Tạo nhánh mới cho tính năng của bạn
3. Thực hiện các thay đổi
4. Gửi pull request

## Giấy Phép

Dự án này được cấp phép theo giấy phép MIT - xem file LICENSE để biết thêm chi tiết.