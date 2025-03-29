# Ứng Dụng Giải Tích và Học Máy

Bộ ví dụ minh họa về các phép toán vector và ứng dụng trong học máy, tập trung vào cost function và gradient descent.

## Cấu Trúc Dự Án

```
calculus-machine-learning/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   └── calculus_ml/
│       ├── __init__.py
│       ├── core/
│       │   ├── vector.py
│       │   ├── linear_model.py
│       │   ├── cost_function.py
│       │   └── gradient_descent.py
│       └── visualization/
│           ├── cost_plot.py
│           └── gradient_plot.py
└── images/
    └── ...
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

3. **Gradient Descent**
   - Tính gradient của cost function
   - Cài đặt thuật toán gradient descent
   - Trực quan hóa quá trình tối ưu

## Chi Tiết Các Hàm Chính

### 1. Vector Operations (`vector.py`)

```python
def dot_product(v1, v2):
    """
    Tính tích vô hướng của hai vector
    Ví dụ: dot_product([1, 2], [3, 4]) -> 11
    """
    return sum(x * y for x, y in zip(v1, v2))

def vector_norm(v):
    """
    Tính độ dài (norm) của vector
    Ví dụ: vector_norm([3, 4]) -> 5.0
    """
    return np.sqrt(sum(x * x for x in v))

def normalize_vector(v):
    """
    Chuẩn hóa vector thành vector đơn vị
    Ví dụ: normalize_vector([3, 4]) -> [0.6, 0.8]
    """
    norm = vector_norm(v)
    return [x / norm for x in v]
```

### 2. Cost Function (`cost_function.py`)

```python
def compute_cost(x, y, w, b):
    """
    Tính cost function J(w,b)
    J(w,b) = (1/2m) * Σ(f(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
    
    Args:
        x (ndarray): training data
        y (ndarray): target values
        w,b (scalar): model parameters
    
    Returns:
        cost (scalar): giá trị của cost function
    """
    m = len(x)
    total_cost = sum((w * x[i] + b - y[i]) ** 2 for i in range(m))
    return total_cost / (2 * m)
```

### 3. Gradient Descent (`gradient_descent.py`)

```python
def compute_gradient(x, y, w, b):
    """
    Tính gradient của cost function J(w,b)
    
    Args:
        x,y: training data
        w,b: model parameters
    
    Returns:
        dj_dw: gradient theo w
        dj_db: gradient theo b
    """
    m = len(x)
    dj_dw = sum((w * x[i] + b - y[i]) * x[i] for i in range(m)) / m
    dj_db = sum(w * x[i] + b - y[i] for i in range(m)) / m
    return dj_dw, dj_db

def gradient_descent(x, y, w_init, b_init, alpha, num_iters):
    """
    Thực hiện gradient descent để tối ưu w,b
    
    Args:
        x,y: training data
        w_init,b_init: giá trị khởi tạo
        alpha: learning rate
        num_iters: số iteration
    
    Returns:
        w,b: tham số tối ưu
        J_history: lịch sử cost
        p_history: lịch sử tham số
    """
```

## Minh Họa Trực Quan

1. **Cost Function 3D**
![Cost Function 3D](images/cost_function_3d.png)
- Bề mặt cost function J(w,b)
- Điểm thấp nhất: tham số tối ưu (w*, b*)

2. **Gradient Descent**
![Gradient Descent](images/gradient_descent_3d.png)
- Đường màu đỏ: quá trình tối ưu
- Điểm đỏ: vị trí bắt đầu
- Điểm xanh: vị trí kết thúc

3. **Cost History**
![Cost History](images/cost_history.png)
- Sự hội tụ của cost function
- Ảnh hưởng của learning rate

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