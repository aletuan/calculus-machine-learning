import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from ..core.logistic.regression import LogisticRegression
from ..visualization.base.plot_utils import setup_plot, save_plot, create_meshgrid, PlotConfig

console = Console()

def generate_admission_data():
    """Tạo dữ liệu mẫu về tuyển sinh"""
    np.random.seed(42)
    n_samples = 100
    
    # Điểm thi và GPA
    x1 = np.random.normal(65, 15, n_samples)  # điểm thi (0-100)
    x2 = np.random.normal(3.0, 0.8, n_samples)  # GPA (0-4)
    
    # Giới hạn giá trị
    x1 = np.clip(x1, 0, 100)
    x2 = np.clip(x2, 0, 4)
    
    # Tạo nhãn dựa trên quy tắc thực tế
    z = 0.05*x1 + 0.8*x2 - 4.5
    prob = 1 / (1 + np.exp(-z))
    y = (prob > 0.5).astype(int)
    
    # Đảm bảo có cả sinh viên đỗ và trượt
    while np.sum(y) == 0 or np.sum(y) == n_samples:
        z = 0.05*x1 + 0.8*x2 - 4.5
        prob = 1 / (1 + np.exp(-z))
        y = (prob > 0.5).astype(int)
    
    return x1, x2, y

def plot_logistic_results(model, X1, X2, y, history, save_dir='images'):
    """Vẽ kết quả logistic regression"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Decision boundary
    setup_plot('Logistic Regression Decision Boundary', 'Test Score', 'GPA')
    
    # Vẽ decision boundary với contour fill
    xx1, xx2 = create_meshgrid(X1, X2, n_points=200)  # Tăng số điểm lưới
    X_grid = np.c_[xx1.ravel(), xx2.ravel()]
    Z = model.predict(X_grid)
    Z = Z.reshape(xx1.shape)
    
    # Vẽ contour fill để thể hiện xác suất
    contour = plt.contourf(xx1, xx2, Z, levels=np.linspace(0, 1, 11), alpha=0.3, cmap='RdYlBu')
    plt.colorbar(contour, label='P(Admitted)')
    
    # Vẽ decision boundary với độ dày tăng
    decision_boundary = plt.contour(xx1, xx2, Z, levels=[0.5], colors='green', linewidths=2, linestyles='--')
    
    # Tạo proxy artist cho decision boundary trong legend
    from matplotlib.lines import Line2D
    proxy = Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Decision Boundary')
    
    # Vẽ scatter plot các điểm dữ liệu
    plt.scatter(X1[y==0], X2[y==0], color='red', alpha=0.7, label='Rejected', edgecolors='k')
    plt.scatter(X1[y==1], X2[y==1], color='blue', alpha=0.7, label='Admitted', edgecolors='k')
    
    # Thêm legend với proxy artist
    plt.legend(handles=[proxy, plt.scatter([], [], color='red', alpha=0.7, label='Rejected', edgecolors='k'),
                       plt.scatter([], [], color='blue', alpha=0.7, label='Admitted', edgecolors='k')])
    
    save_plot(os.path.join(save_dir, 'logistic_decision_boundary.png'))
    
    # Plot 2: Cost history
    setup_plot('Cost Function History', 'Iteration', 'Cost J(w₁,w₂,b)')
    plt.plot(history['cost_history'])
    save_plot(os.path.join(save_dir, 'logistic_cost_history.png'))

def run_logistic_example():
    """Chạy ví dụ logistic regression"""
    console.print("\n[bold cyan]Logistic Regression Example[/bold cyan]", justify="center")
    
    # Hiển thị công thức
    console.print(Panel(
        "[bold green]Công thức tính toán:[/bold green]\n"
        "1. Hàm sigmoid: g(z) = 1 / (1 + e^(-z))\n"
        "2. Hàm dự đoán: h(x) = g(w₁x₁ + w₂x₂ + b)\n"
        "3. Cost function: J(w₁,w₂,b) = -(1/m) * Σ[y⁽ⁱ⁾log(h(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h(x⁽ⁱ⁾))]\n"
        "4. Gradient:\n"
        "   - ∂J/∂w₁ = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x₁⁽ⁱ⁾\n"
        "   - ∂J/∂w₂ = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x₂⁽ⁱ⁾\n"
        "   - ∂J/∂b = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)",
        title="Logistic Regression Formulas",
        border_style="cyan"
    ))
    
    # Tạo dữ liệu
    X1, X2, y = generate_admission_data()
    X = np.column_stack((X1, X2))
    
    # Hiển thị thông tin dữ liệu
    console.print(Panel(
        f"[bold green]Dữ liệu tuyển sinh:[/bold green]\n"
        f"- Số lượng mẫu: {len(y)}\n"
        f"- Số lượng sinh viên đỗ: {np.sum(y)}\n"
        f"- Số lượng sinh viên trượt: {len(y) - np.sum(y)}\n"
        f"- Điểm thi: min={X1.min():.1f}, max={X1.max():.1f}, mean={X1.mean():.1f}\n"
        f"- GPA: min={X2.min():.1f}, max={X2.max():.1f}, mean={X2.mean():.1f}",
        title="Training Data",
        border_style="cyan"
    ))
    
    # Khởi tạo model
    model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    
    # Hiển thị quá trình training
    with Progress() as progress:
        task = progress.add_task("[cyan]Training...", total=model.num_iterations)
        
        def callback(epoch, cost):
            progress.update(task, advance=1)
            if epoch % 100 == 0 or epoch == model.num_iterations - 1:
                console.print(f"[yellow]Epoch {epoch + 1}/{model.num_iterations}[/yellow] - Loss: {cost:.4f}")
        
        # Training model với callback
        history = model.fit(X, y, callback=callback)
    
    # Hiển thị kết quả
    console.print(Panel(
        f"[bold green]Kết quả tối ưu:[/bold green]\n"
        f"1. Tham số tối ưu:\n"
        f"   w₁ = {model.weights[0]:.4f} (trọng số điểm thi)\n"
        f"   w₂ = {model.weights[1]:.4f} (trọng số GPA)\n"
        f"   b = {model.bias:.4f} (độ chệch)\n"
        f"2. Phương trình phân loại:\n"
        f"   P(đỗ) = g({model.weights[0]:.4f}x₁ + {model.weights[1]:.4f}x₂ + {model.bias:.4f})\n"
        f"3. Cost cuối cùng: {history['cost_history'][-1]:.4f}",
        title="Optimization Results",
        border_style="cyan"
    ))
    
    # Vẽ kết quả
    plot_logistic_results(model, X1, X2, y, history)
    console.print("[green]✓[/green] Logistic regression visualization saved")

if __name__ == "__main__":
    run_logistic_example() 