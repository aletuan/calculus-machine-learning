import numpy as np

def sigmoid(z):
    """
    Hàm sigmoid: g(z) = 1 / (1 + e^(-z))
    """
    return 1 / (1 + np.exp(-z))

def predict_logistic(x1, x2, w1, w2, b):
    """
    Hàm dự đoán xác suất: P(y=1) = g(w₁x₁ + w₂x₂ + b)
    """
    z = w1*x1 + w2*x2 + b
    return sigmoid(z)

def compute_cost_logistic(x1, x2, y, w1, w2, b):
    """
    Cost function cho logistic regression:
    J(w₁,w₂,b) = -(1/m) * Σ[y⁽ⁱ⁾log(h(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h(x⁽ⁱ⁾))]
    """
    m = len(y)
    predictions = predict_logistic(x1, x2, w1, w2, b)
    
    # Tránh log(0)
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
    cost = -(1/m) * np.sum(y*np.log(predictions) + (1-y)*np.log(1-predictions))
    return cost

def compute_gradient_logistic(x1, x2, y, w1, w2, b):
    """
    Tính gradient cho logistic regression:
    ∂J/∂w₁ = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x₁⁽ⁱ⁾
    ∂J/∂w₂ = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x₂⁽ⁱ⁾
    ∂J/∂b = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)
    """
    m = len(y)
    predictions = predict_logistic(x1, x2, w1, w2, b)
    error = predictions - y
    
    dj_dw1 = (1/m) * np.sum(error * x1)
    dj_dw2 = (1/m) * np.sum(error * x2)
    dj_db = (1/m) * np.sum(error)
    
    return dj_dw1, dj_dw2, dj_db

def gradient_descent_logistic(x1, x2, y, w1_init, w2_init, b_init, alpha, num_iters):
    """
    Thực hiện gradient descent để tối ưu w₁, w₂, b
    """
    w1 = w1_init
    w2 = w2_init
    b = b_init
    J_history = []
    p_history = []
    
    for i in range(num_iters):
        # Tính gradient
        dj_dw1, dj_dw2, dj_db = compute_gradient_logistic(x1, x2, y, w1, w2, b)
        
        # Cập nhật tham số
        w1 = w1 - alpha * dj_dw1
        w2 = w2 - alpha * dj_dw2
        b = b - alpha * dj_db
        
        # Lưu lại lịch sử
        if i < 100000:
            J_history.append(compute_cost_logistic(x1, x2, y, w1, w2, b))
            p_history.append([w1, w2, b])
    
    return w1, w2, b, J_history, p_history

def generate_admission_data():
    """
    Tạo dữ liệu mẫu về tuyển sinh:
    - x₁: điểm thi (0-100)
    - x₂: GPA (0-4)
    - y: kết quả (0: trượt, 1: đỗ)
    """
    np.random.seed(42)
    n_samples = 100
    
    # Tạo điểm thi và GPA với phân phối cân bằng hơn
    x1 = np.random.normal(65, 15, n_samples)  # điểm thi (0-100)
    x2 = np.random.normal(3.0, 0.8, n_samples)  # GPA (0-4)
    
    # Giới hạn giá trị trong khoảng hợp lý
    x1 = np.clip(x1, 0, 100)
    x2 = np.clip(x2, 0, 4)
    
    # Tạo nhãn dựa trên quy tắc thực tế hơn
    # Điểm thi quan trọng hơn GPA
    z = 0.05*x1 + 0.8*x2 - 4.5
    prob = sigmoid(z)
    y = (prob > 0.5).astype(int)
    
    # Đảm bảo có cả sinh viên đỗ và trượt
    while np.sum(y) == 0 or np.sum(y) == n_samples:
        z = 0.05*x1 + 0.8*x2 - 4.5
        prob = sigmoid(z)
        y = (prob > 0.5).astype(int)
    
    return x1, x2, y 