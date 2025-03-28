import numpy as np

def create_vectors():
    """Tạo các vector mẫu"""
    v1 = np.array([3, 4])  # Vector [3, 4]
    v2 = np.array([2, 1])  # Vector [2, 1]
    return v1, v2

def vector_operations(v1, v2):
    """Thực hiện các phép toán vector cơ bản"""
    # Cộng vector
    v_sum = v1 + v2
    
    # Trừ vector
    v_diff = v1 - v2
    
    # Nhân vector với một số
    scalar = 2
    v_scaled = scalar * v1
    
    # Độ dài (norm) của vector
    v1_norm = np.linalg.norm(v1)
    
    # Tích vô hướng (dot product)
    dot_product = np.dot(v1, v2)
    
    # Tích có hướng (cross product) - chỉ trong không gian 3D
    v1_3d = np.array([v1[0], v1[1], 0])
    v2_3d = np.array([v2[0], v2[1], 0])
    cross_product = np.cross(v1_3d, v2_3d)
    
    return {
        'v_sum': v_sum,
        'v_diff': v_diff,
        'v_scaled': v_scaled,
        'scalar': scalar,
        'v1_norm': v1_norm,
        'dot_product': dot_product,
        'cross_product': cross_product
    }

def unit_vector_and_angle(v1, v2):
    """Tính vector đơn vị và góc giữa hai vector"""
    # Vector đơn vị của v1
    v1_unit = v1 / np.linalg.norm(v1)
    v1_unit_norm = np.linalg.norm(v1_unit)
    
    # Góc giữa hai vector
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cos_theta) * 180 / np.pi
    
    return {
        'v1_unit': v1_unit,
        'v1_unit_norm': v1_unit_norm,
        'angle': angle,
        'cos_theta': cos_theta
    }

def vector_projection(v1, v2):
    """Tính chiếu vector v1 lên v2"""
    projection_scalar = np.dot(v1, v2) / np.dot(v2, v2)
    projection_vector = projection_scalar * v2
    projection_norm = np.linalg.norm(projection_vector)
    
    return {
        'projection_vector': projection_vector,
        'projection_norm': projection_norm
    } 