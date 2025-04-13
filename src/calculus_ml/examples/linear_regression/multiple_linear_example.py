import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from src.calculus_ml.core.linear.regression import LinearRegression
from src.calculus_ml.core.base.scaler import StandardScaler
from src.calculus_ml.visualization.base.plot_utils import setup_plot, save_plot, PlotConfig

console = Console()

def generate_house_data(n_samples=100, noise=0.1):
    """Tạo dữ liệu mẫu cho bài toán dự đoán giá nhà"""
    np.random.seed(42)
    # Diện tích (1000 sqft)
    area = np.random.normal(2.5, 1, n_samples)
    # Số phòng ngủ
    bedrooms = np.random.randint(1, 6, n_samples)
    # Giá nhà (1000$)
    price = 200*area + 50*bedrooms + 100 + np.random.normal(0, noise, n_samples)
    return area, bedrooms, price

def plot_multiple_results(model, X1, X2, y, history, scaler=None, save_dir='images'):
    """Vẽ kết quả multiple linear regression"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Dữ liệu và mặt phẳng hồi quy
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Vẽ dữ liệu
    ax.scatter(X1, X2, y, color='blue', alpha=0.7, label='Data points')
    
    # Vẽ mặt phẳng hồi quy
    x1_min, x1_max = X1.min() - 1, X1.max() + 1
    x2_min, x2_max = X2.min() - 1, X2.max() + 1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 10),
                          np.linspace(x2_min, x2_max, 10))
    X_grid = np.column_stack((xx1.ravel(), xx2.ravel()))
    
    if scaler is not None:
        X_grid_scaled = scaler.transform(X_grid)
        y_pred = model.predict(X_grid_scaled)
    else:
        y_pred = model.predict(X_grid)
    
    y_pred = y_pred.reshape(xx1.shape)
    
    ax.plot_surface(xx1, xx2, y_pred, alpha=0.3, color='red', label='Regression plane')
    
    ax.set_xlabel('Area (1000 sqft)')
    ax.set_ylabel('Bedrooms')
    ax.set_zlabel('Price (1000$)')
    ax.set_title('Multiple Linear Regression')
    ax.legend()
    
    save_plot(os.path.join(save_dir, 'multiple_regression_fit.png'))
    
    # Plot 2: Cost history
    setup_plot('Cost Function History', 'Iteration', 'Cost J(w₁,w₂,b)')
    plt.plot(history['cost_history'])
    save_plot(os.path.join(save_dir, 'multiple_cost_history.png'))

def run_multiple_example():
    """Chạy ví dụ multiple linear regression"""
    console.print("\n[bold cyan]Multiple Linear Regression Example[/bold cyan]", justify="center")
    
    # Hiển thị công thức
    console.print(Panel(
        "[bold green]Công thức tính toán:[/bold green]\n"
        "1. Hàm dự đoán: h(x) = w₁x₁ + w₂x₂ + b\n"
        "2. Cost function: J(w₁,w₂,b) = (1/2m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²\n"
        "3. Gradient:\n"
        "   - ∂J/∂w₁ = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x₁⁽ⁱ⁾\n"
        "   - ∂J/∂w₂ = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x₂⁽ⁱ⁾\n"
        "   - ∂J/∂b = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)\n"
        "4. Feature Scaling:\n"
        "   - Chuẩn hóa dữ liệu: x' = (x - μ) / σ\n"
        "   - μ: mean, σ: standard deviation",
        title="Multiple Regression Formulas",
        border_style="cyan"
    ))
    
    # Tạo dữ liệu
    area, bedrooms, price = generate_house_data()
    X = np.column_stack((area, bedrooms))
    
    # Hiển thị thông tin dữ liệu gốc
    console.print(Panel(
        f"[bold green]Dữ liệu gốc:[/bold green]\n"
        f"- Số lượng mẫu: {len(price)}\n"
        f"- Diện tích: min={area.min():.1f}, max={area.max():.1f}, mean={area.mean():.1f}\n"
        f"- Số phòng ngủ: min={bedrooms.min()}, max={bedrooms.max()}, mean={bedrooms.mean():.1f}\n"
        f"- Giá nhà: min={price.min():.1f}, max={price.max():.1f}, mean={price.mean():.1f}",
        title="Original Data",
        border_style="cyan"
    ))
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Hiển thị thông tin dữ liệu đã chuẩn hóa
    console.print(Panel(
        f"[bold green]Dữ liệu đã chuẩn hóa:[/bold green]\n"
        f"- Diện tích: min={X_scaled[:,0].min():.2f}, max={X_scaled[:,0].max():.2f}, mean={X_scaled[:,0].mean():.2f}\n"
        f"- Số phòng ngủ: min={X_scaled[:,1].min():.2f}, max={X_scaled[:,1].max():.2f}, mean={X_scaled[:,1].mean():.2f}",
        title="Scaled Data",
        border_style="cyan"
    ))
    
    # Hiển thị quá trình training
    console.print("\n[bold yellow]Quá trình training:[/bold yellow]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Training...", total=1000)
        
        def training_callback(epoch, cost):
            progress.update(task, advance=1)
            if epoch % 100 == 0 or epoch == 999:
                console.print(f"Epoch {epoch:4d}: Cost = {cost:.4f}")
        
        # Khởi tạo và training model
        model = LinearRegression(learning_rate=0.01, num_iterations=1000)
        history = model.fit(X_scaled, price, callback=training_callback)
    
    # Hiển thị kết quả
    results = [
        "[bold green]Kết quả tối ưu:[/bold green]",
        "1. Tham số tối ưu:",
        f"   w₁ = {model.weights[0]:.4f} (trọng số diện tích)",
        f"   w₂ = {model.weights[1]:.4f} (trọng số số phòng)",
        f"   b = {model.bias:.4f} (độ chệch)",
        "2. Phương trình hồi quy:",
        f"   Giá = {model.weights[0]:.4f}*Diện_tích + {model.weights[1]:.4f}*Số_phòng + {model.bias:.4f}",
        f"3. Cost cuối cùng: {history['cost_history'][-1]:.4f}"
    ]
    
    console.print(Panel(
        "\n".join(results),
        title="Optimization Results",
        border_style="cyan"
    ))
    
    # Vẽ kết quả
    plot_multiple_results(model, area, bedrooms, price, history, scaler)
    console.print("[green]✓[/green] Multiple regression visualization saved")

if __name__ == "__main__":
    run_multiple_example() 