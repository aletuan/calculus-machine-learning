import numpy as np
from .cost_function import predict, predict_2d

def compute_gradient(x, y, w, b):
    """Tính gradient của cost function J(w,b) cho 1 feature"""
    m = len(x)
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        y_pred = predict(x[i], w, b)
        error = y_pred - y[i]
        dj_dw += error * x[i]
        dj_db += error
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw, dj_db

def compute_gradient_2d(x1, x2, y, w1, w2, b):
    """Tính gradient của cost function J(w1,w2,b) cho 2 features"""
    m = len(x1)
    dj_dw1 = 0
    dj_dw2 = 0
    dj_db = 0
    
    for i in range(m):
        y_pred = predict_2d(x1[i], x2[i], w1, w2, b)
        error = y_pred - y[i]
        dj_dw1 += error * x1[i]
        dj_dw2 += error * x2[i]
        dj_db += error
    
    dj_dw1 = dj_dw1 / m
    dj_dw2 = dj_dw2 / m
    dj_db = dj_db / m
    
    return dj_dw1, dj_dw2, dj_db

def gradient_descent(x, y, w_init, b_init, alpha, num_iters, compute_cost):
    """Thực hiện gradient descent để tối ưu w,b cho 1 feature"""
    w = w_init
    b = b_init
    J_history = []
    p_history = []
    
    for i in range(num_iters):
        # Tính gradient
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        
        # Cập nhật tham số
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # Lưu lại lịch sử
        if i < 100000:
            J_history.append(compute_cost(x, y, w, b))
            p_history.append([w, b])
    
    return w, b, J_history, p_history

def gradient_descent_2d(x1, x2, y, w1_init, w2_init, b_init, alpha, num_iters, compute_cost_2d):
    """Thực hiện gradient descent để tối ưu w1,w2,b cho 2 features"""
    w1 = w1_init
    w2 = w2_init
    b = b_init
    J_history = []
    p_history = []
    
    for i in range(num_iters):
        # Tính gradient
        dj_dw1, dj_dw2, dj_db = compute_gradient_2d(x1, x2, y, w1, w2, b)
        
        # Cập nhật tham số
        w1 = w1 - alpha * dj_dw1
        w2 = w2 - alpha * dj_dw2
        b = b - alpha * dj_db
        
        # Lưu lại lịch sử
        if i < 100000:
            J_history.append(compute_cost_2d(x1, x2, y, w1, w2, b))
            p_history.append([w1, w2, b])
    
    return w1, w2, b, J_history, p_history 