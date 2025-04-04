import matplotlib.pyplot as plt
import numpy as np

class PlotConfig:
    """Cấu hình chung cho việc vẽ"""
    COLORS = ['blue', 'red', 'green', 'purple', 'orange']
    FIGSIZE = (10, 6)
    DPI = 100
    FONT_SIZE = 12

def setup_plot(title, xlabel, ylabel, figsize=None):
    """Khởi tạo plot với các thông số cơ bản"""
    if figsize is None:
        figsize = PlotConfig.FIGSIZE
        
    plt.figure(figsize=figsize, dpi=PlotConfig.DPI)
    plt.title(title, fontsize=PlotConfig.FONT_SIZE)
    plt.xlabel(xlabel, fontsize=PlotConfig.FONT_SIZE)
    plt.ylabel(ylabel, fontsize=PlotConfig.FONT_SIZE)
    plt.grid(True)

def save_plot(filename, tight_layout=True):
    """Lưu plot vào file"""
    if tight_layout:
        plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=PlotConfig.DPI)
    plt.close()

def create_meshgrid(x1, x2, n_points=100):
    """Tạo lưới điểm cho contour plot"""
    x1_min, x1_max = x1.min()-1, x1.max()+1
    x2_min, x2_max = x2.min()-1, x2.max()+1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, n_points),
                          np.linspace(x2_min, x2_max, n_points))
    return xx1, xx2 