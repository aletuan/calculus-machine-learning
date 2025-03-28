import numpy as np

def compute_gradient(x, y, w, b):
    """
    Tính gradient của cost function J(w,b)
    
    Args:
        x (ndarray (m,)): Data, m examples 
        y (ndarray (m,)): target values
        w,b (scalar)    : model parameters
    
    Returns:
        dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
        dj_db (scalar): The gradient of the cost w.r.t. the parameter b
    """
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        # Gradient của w
        dj_dw_i = (f_wb - y[i]) * x[i]
        # Gradient của b
        dj_db_i = f_wb - y[i]
        
        # Cộng dồn
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    
    # Chia trung bình
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw, dj_db

def gradient_descent(x, y, w_init, b_init, alpha, num_iters, compute_cost):
    """
    Thực hiện gradient descent để tối ưu w,b
    
    Args:
        x (ndarray (m,))  : Data, m examples
        y (ndarray (m,))  : target values
        w_init,b_init (scalar) : initial values of model parameters  
        alpha (float)     : Learning rate
        num_iters (int)   : number of iterations
        compute_cost      : function to compute cost
    
    Returns:
        w (scalar)        : Updated value of parameter after running gradient descent
        b (scalar)        : Updated value of parameter after running gradient descent
        J_history         : History of cost values
        p_history         : History of parameters [w,b] 
    """
    # Khởi tạo giá trị
    w = w_init
    b = b_init
    
    # Mảng lưu lịch sử để vẽ đồ thị
    J_history = []
    p_history = []
    
    for i in range(num_iters):
        # Tính gradient
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        
        # Cập nhật tham số
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # Lưu lịch sử
        if i < 100000:  # Ngăn tràn bộ nhớ
            J_history.append(compute_cost(x, y, w, b))
            p_history.append([w, b])
        
        # In thông tin mỗi 10% số iteration
        if i % (num_iters // 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    
    return w, b, J_history, p_history 