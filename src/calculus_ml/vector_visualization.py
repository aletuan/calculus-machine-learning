import matplotlib.pyplot as plt
import numpy as np

def setup_plot(figsize=(10, 6)):
    """Thiết lập cơ bản cho đồ thị"""
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=figsize)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(alpha=0.3)

def plot_basic_vectors(v1, v2, v_sum, v_diff):
    """Vẽ các vector cơ bản và phép toán"""
    setup_plot()
    
    # Vẽ các vector bắt đầu từ gốc tọa độ
    plt.arrow(0, 0, v1[0], v1[1], head_width=0.2, head_length=0.3, 
             fc='blue', ec='blue', label='v1 = [3, 4]')
    plt.arrow(0, 0, v2[0], v2[1], head_width=0.2, head_length=0.3, 
             fc='red', ec='red', label='v2 = [2, 1]')
    
    # Vẽ tổng và hiệu vector
    plt.arrow(0, 0, v_sum[0], v_sum[1], head_width=0.2, head_length=0.3, 
             fc='green', ec='green', label='v1 + v2 = [5, 5]')
    plt.arrow(0, 0, v_diff[0], v_diff[1], head_width=0.2, head_length=0.3, 
             fc='purple', ec='purple', label='v1 - v2 = [1, 3]')
    
    plt.xlim(-1, 6)
    plt.ylim(-1, 6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Biểu diễn vector trong không gian 2D')
    plt.legend()
    plt.savefig('images/vectors_basic.png')
    plt.close()

def plot_vector_angle(v1, v2, angle, cos_theta):
    """Vẽ góc giữa hai vector"""
    setup_plot(figsize=(8, 6))
    
    plt.arrow(0, 0, v1[0], v1[1], head_width=0.2, head_length=0.3, 
             fc='blue', ec='blue', label='v1 = [3, 4]')
    plt.arrow(0, 0, v2[0], v2[1], head_width=0.2, head_length=0.3, 
             fc='red', ec='red', label='v2 = [2, 1]')
    
    # Vẽ cung tròn để hiển thị góc
    radius = 1
    theta = np.linspace(0, np.arccos(cos_theta), 100)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    plt.plot(x, y, 'g-', alpha=0.7)
    plt.annotate(f'{angle:.1f}°', xy=(radius/2, radius/4), fontsize=12)
    
    plt.xlim(-1, 5)
    plt.ylim(-1, 5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Góc giữa hai vector')
    plt.legend()
    plt.savefig('images/vector_angle.png')
    plt.close()

def plot_vector_projection(v1, v2, projection_vector):
    """Vẽ phép chiếu vector"""
    setup_plot()
    
    # Vẽ các vector gốc
    plt.arrow(0, 0, v1[0], v1[1], head_width=0.2, head_length=0.3, 
             fc='blue', ec='blue', label='v1 = [3, 4]')
    plt.arrow(0, 0, v2[0], v2[1], head_width=0.2, head_length=0.3, 
             fc='red', ec='red', label='v2 = [2, 1]')
    
    # Vẽ vector chiếu
    plt.arrow(0, 0, projection_vector[0], projection_vector[1], 
             head_width=0.2, head_length=0.3, fc='green', ec='green', 
             label='proj_v1_onto_v2')
    
    # Vẽ đường từ v1 đến vector chiếu
    plt.plot([v1[0], projection_vector[0]], [v1[1], projection_vector[1]], 
             'k--', alpha=0.7)
    
    plt.xlim(-1, 5)
    plt.ylim(-1, 5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Chiếu vector v1 lên v2')
    plt.legend()
    plt.savefig('images/vector_projection.png')
    plt.close() 