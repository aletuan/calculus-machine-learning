import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
from ..core.linear.regression import LinearRegression
from ..visualization.base.plot_utils import setup_plot, save_plot, PlotConfig

console = Console()

def generate_house_data():
    """Tạo dữ liệu mẫu về giá nhà"""
    np.random.seed(42)
    n_samples = 100
    
    # Kích thước nhà (1000 sqft)
    x = np.random.normal(2.5, 1.0, n_samples)
    x = np.clip(x, 1, 5)
    
    # Giá nhà (1000s $)
    y = 200 * x + 100 + np.random.normal(0, 50, n_samples)
    y = np.clip(y, 100, 1000)
    
    return x, y

def plot_linear_results(model, X, y, history, save_dir='images'):
    """Vẽ kết quả linear regression"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Dữ liệu và đường hồi quy
    setup_plot('Linear Regression Fit', 'House Size (1000 sqft)', 'Price (1000s $)')
    plt.scatter(X, y, color='blue', alpha=0.5, label='Data')
    
    x_range = np.linspace(X.min(), X.max(), 100)
    y_pred = model.predict(x_range)
    plt.plot(x_range, y_pred, 'r-', label=f'Fit: y = {model.weights[0]:.2f}x + {model.bias:.2f}')
    
    plt.legend()
    save_plot(os.path.join(save_dir, 'linear_regression_fit.png'))
    
    # Plot 2: Cost history
    setup_plot('Cost Function History', 'Iteration', 'Cost J(w,b)')
    plt.plot(history['cost_history'])
    save_plot(os.path.join(save_dir, 'linear_cost_history.png'))

def run_linear_example():
    """Chạy ví dụ linear regression"""
    console.print("\n[bold cyan]Linear Regression Example[/bold cyan]", justify="center")
    
    # Hiển thị công thức
    console.print(Panel(
        "[bold green]Công thức tính toán:[/bold green]\n"
        "1. Hàm dự đoán: f(x) = wx + b\n"
        "2. Cost function: J(w,b) = (1/2m) * Σ(f(x⁽ⁱ⁾) - y⁽ⁱ⁾)²\n"
        "3. Gradient:\n"
        "   - ∂J/∂w = (1/m) * Σ(f(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x⁽ⁱ⁾\n"
        "   - ∂J/∂b = (1/m) * Σ(f(x⁽ⁱ⁾) - y⁽ⁱ⁾)",
        title="Linear Regression Formulas",
        border_style="cyan"
    ))
    
    # Tạo dữ liệu
    X, y = generate_house_data()
    
    # Hiển thị thông tin dữ liệu
    console.print(Panel(
        f"[bold green]Dữ liệu mẫu:[/bold green]\n"
        f"- Số mẫu: {len(y)}\n"
        f"- Kích thước nhà: min={X.min():.1f}, max={X.max():.1f}, mean={X.mean():.1f}\n"
        f"- Giá nhà: min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.1f}",
        title="Training Data",
        border_style="cyan"
    ))
    
    # Khởi tạo và training model
    model = LinearRegression(learning_rate=0.01, num_iterations=1000)
    history = model.fit(X, y)
    
    # Hiển thị kết quả
    console.print(Panel(
        f"[bold green]Kết quả tối ưu:[/bold green]\n"
        f"1. Tham số tối ưu:\n"
        f"   w = {model.weights[0]:.2f}\n"
        f"   b = {model.bias:.2f}\n"
        f"2. Phương trình hồi quy:\n"
        f"   y = {model.weights[0]:.2f}x + {model.bias:.2f}\n"
        f"3. Cost cuối cùng: {history['cost_history'][-1]:.4f}",
        title="Optimization Results",
        border_style="cyan"
    ))
    
    # Vẽ kết quả
    plot_linear_results(model, X, y, history)
    console.print("[green]✓[/green] Linear regression visualization saved")

if __name__ == "__main__":
    run_linear_example() 