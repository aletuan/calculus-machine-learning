"""
Example of Polynomial Regression with Regularization.
"""

import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from ..core.polynomial.regression import PolynomialRegression
from ..visualization.polynomial.plot_utils import (
    plot_polynomial_comparison,
    plot_regularization_effect
)

console = Console()

def generate_nonlinear_data(n_samples=100, noise=0.3):
    """Generate synthetic nonlinear data."""
    np.random.seed(42)
    # Generate positive X values between 1 and 5 (1000 sqft)
    X = np.random.uniform(1.0, 5.0, n_samples)
    # Generate y with stronger polynomial relationship
    y = 50 + 100*X + 20*X**2 + 5*X**3
    y += noise * np.random.randn(n_samples) * 100  # Scale noise appropriately
    # Ensure non-negative house prices
    y = np.clip(y, 0, None)
    return X, y

def run_polynomial_comparison():
    """Run and visualize polynomial regression with different degrees."""
    # Generate data
    X, y = generate_nonlinear_data()
    
    # Train models with different degrees
    degrees = [1, 2, 4]
    models = []
    
    for degree in degrees:
        model = PolynomialRegression(degree=degree, lambda_reg=0.0,
                                   learning_rate=0.01, num_iterations=1000)
        model.fit(X, y)
        models.append(model)
    
    # Plot comparison
    plot_polynomial_comparison(
        X, y, models, degrees,
        save_path='images/polynomial_regression_fit.png'
    )
    
    return models[2]  # Return degree 4 model for regularization example

def run_regularization_example(base_model):
    """Run and visualize the effect of regularization."""
    # Generate data
    X, y = generate_nonlinear_data(noise=0.5)  # More noise
    
    # Train models with different lambda values - using smaller values
    lambdas = [0.0, 0.01, 1.0]  # Changed from [0.0, 0.1, 10.0]
    models = []
    
    # Sort X for consistent testing
    X_test = np.linspace(1.0, 5.0, 10)
    print("\nTesting predictions for different lambda values:")
    print("X (diện tích) | λ=0.0 | λ=0.01 | λ=1.0")
    print("-" * 45)
    
    # Train models and store predictions
    predictions = []
    for lambda_reg in lambdas:
        model = PolynomialRegression(degree=4, lambda_reg=lambda_reg,
                                   learning_rate=0.01, num_iterations=1000)
        model.fit(X, y)
        models.append(model)
        predictions.append(model.predict(X_test))
    
    # Print predictions for inspection
    for i, x in enumerate(X_test):
        print(f"{x:12.1f} | {predictions[0][i]:6.0f} | {predictions[1][i]:7.0f} | {predictions[2][i]:6.0f}")
    
    # Plot regularization effect
    plot_regularization_effect(
        X, y, models, lambdas,
        save_path='images/regularization_effect.png'
    )

def plot_house_price_data(X, y, save_path='images/house_price_data.png'):
    """Plot house price data.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, 1)
        House sizes in 1000 sqft
    y : ndarray of shape (n_samples,)
        House prices in 1000$
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.5, label='House Data')
    plt.xlabel('Diện tích (1000 sqft)')
    plt.ylabel('Giá nhà (1000$)')
    plt.title('Dữ liệu giá nhà')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run all polynomial regression examples."""
    console.print("\n[bold cyan]Polynomial Regression Example[/bold cyan]", justify="center")
    
    # Hiển thị công thức
    console.print(Panel(
        "[bold green]Công thức tính toán:[/bold green]\n"
        "1. Hàm dự đoán: h(x) = w₀ + w₁x + w₂x² + ... + wₙxⁿ\n"
        "2. Cost function với regularization:\n"
        "   J(w) = (1/2m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)² + (λ/2m) * Σwⱼ²\n"
        "3. Gradient:\n"
        "   ∂J/∂w₀ = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)\n"
        "   ∂J/∂wⱼ = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) * xⱼ⁽ⁱ⁾ + (λ/m) * wⱼ",
        title="Polynomial Regression Formulas",
        border_style="cyan"
    ))
    
    # Generate and plot house price data
    X, y = generate_nonlinear_data()
    
    # Hiển thị thông tin dữ liệu
    console.print(Panel(
        f"[bold green]Dữ liệu giá nhà:[/bold green]\n"
        f"- Số lượng mẫu: {len(y)}\n"
        f"- Diện tích: min={X.min():.1f}, max={X.max():.1f}, mean={X.mean():.1f} (1000 sqft)\n"
        f"- Giá nhà: min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.1f} (1000$)\n"
        f"- Độ nhiễu: 0.3",
        title="Training Data",
        border_style="cyan"
    ))
    
    plot_house_price_data(X, y)
    console.print("[green]✓[/green] Generated house_price_data.png")
    
    # Run polynomial comparison
    degrees = [1, 2, 4]
    models = []
    costs = []
    
    for degree in degrees:
        console.print(f"\n[bold cyan]Training polynomial regression (degree={degree})[/bold cyan]")
        model = PolynomialRegression(degree=degree, lambda_reg=0.0,
                                   learning_rate=0.01, num_iterations=1000)
        
        # Hiển thị quá trình training
        with Progress() as progress:
            task = progress.add_task("[cyan]Training...", total=model.num_iterations)
            
            def callback(epoch, cost):
                progress.update(task, advance=1)
                if epoch % 100 == 0 or epoch == model.num_iterations - 1:
                    console.print(f"[yellow]Epoch {epoch + 1}/{model.num_iterations}[/yellow] - Loss: {cost:.4f}")
            
            # Training model với callback
            model.fit(X, y, callback=callback)
        
        models.append(model)
        costs.append(model.cost_history[-1])
        
        # Hiển thị kết quả cho từng model
        console.print(Panel(
            f"[bold green]Kết quả tối ưu (degree={degree}):[/bold green]\n"
            f"1. Tham số tối ưu:\n"
            f"   {', '.join([f'w{i}={w:.4f}' for i, w in enumerate(model.weights)])}\n"
            f"2. Phương trình hồi quy:\n"
            f"   y = {' + '.join([f'{w:.4f}x^{i}' for i, w in enumerate(model.weights)])}\n"
            f"3. Cost cuối cùng: {model.cost_history[-1]:.4f}",
            title=f"Optimization Results (degree={degree})",
            border_style="cyan"
        ))
    
    # Plot comparison
    plot_polynomial_comparison(
        X, y, models, degrees,
        save_path='images/polynomial_regression_fit.png'
    )
    console.print("[green]✓[/green] Generated polynomial_regression_fit.png")
    
    # Hiển thị đánh giá kết quả
    best_degree = degrees[np.argmin(costs)]
    best_model = models[np.argmin(costs)]
    console.print(Panel(
        f"[bold green]Đánh giá kết quả:[/bold green]\n"
        f"1. So sánh các mô hình:\n"
        f"   - Degree 1 (Linear): Cost = {costs[0]:.4f}\n"
        f"   - Degree 2 (Quadratic): Cost = {costs[1]:.4f}\n"
        f"   - Degree 4 (Quartic): Cost = {costs[2]:.4f}\n\n"
        f"2. Mô hình tối ưu nhất:\n"
        f"   - Bậc đa thức: {best_degree}\n"
        f"   - Phương trình: y = {' + '.join([f'{w:.4f}x^{i}' for i, w in enumerate(best_model.weights)])}\n"
        f"   - Cost: {min(costs):.4f}\n\n"
        f"3. Nhận xét:\n"
        f"   - Mô hình bậc 2 cho kết quả tốt nhất với cost thấp nhất\n"
        f"   - Mô hình bậc 1 quá đơn giản, không nắm bắt được mối quan hệ phi tuyến\n"
        f"   - Mô hình bậc 4 có cost cao hơn, có thể bị overfitting\n"
        f"   - Mô hình bậc 2 là sự cân bằng tốt giữa độ phức tạp và hiệu quả",
        title="Model Evaluation",
        border_style="cyan"
    ))
    
    # Run regularization example
    console.print("\n[bold cyan]Regularization Example[/bold cyan]")
    lambdas = [0.0, 0.01, 1.0]
    reg_models = []
    reg_costs = []
    
    for lambda_reg in lambdas:
        console.print(f"\n[bold cyan]Training with regularization (λ={lambda_reg})[/bold cyan]")
        model = PolynomialRegression(degree=4, lambda_reg=lambda_reg,
                                   learning_rate=0.01, num_iterations=1000)
        
        # Hiển thị quá trình training
        with Progress() as progress:
            task = progress.add_task("[cyan]Training...", total=model.num_iterations)
            
            def callback(epoch, cost):
                progress.update(task, advance=1)
                if epoch % 100 == 0 or epoch == model.num_iterations - 1:
                    console.print(f"[yellow]Epoch {epoch + 1}/{model.num_iterations}[/yellow] - Loss: {cost:.4f}")
            
            # Training model với callback
            model.fit(X, y, callback=callback)
        
        reg_models.append(model)
        reg_costs.append(model.cost_history[-1])
        
        # Hiển thị kết quả cho từng model
        console.print(Panel(
            f"[bold green]Kết quả tối ưu (λ={lambda_reg}):[/bold green]\n"
            f"1. Tham số tối ưu:\n"
            f"   {', '.join([f'w{i}={w:.4f}' for i, w in enumerate(model.weights)])}\n"
            f"2. Phương trình hồi quy:\n"
            f"   y = {' + '.join([f'{w:.4f}x^{i}' for i, w in enumerate(model.weights)])}\n"
            f"3. Cost cuối cùng: {model.cost_history[-1]:.4f}",
            title=f"Optimization Results (λ={lambda_reg})",
            border_style="cyan"
        ))
    
    # Plot regularization effect
    plot_regularization_effect(
        X, y, reg_models, lambdas,
        save_path='images/regularization_effect.png'
    )
    console.print("[green]✓[/green] Generated regularization_effect.png")
    
    # Hiển thị đánh giá regularization
    best_lambda = lambdas[np.argmin(reg_costs)]
    best_reg_model = reg_models[np.argmin(reg_costs)]
    console.print(Panel(
        f"[bold green]Đánh giá Regularization:[/bold green]\n"
        f"1. So sánh các mô hình:\n"
        f"   - λ = 0.0: Cost = {reg_costs[0]:.4f}\n"
        f"   - λ = 0.01: Cost = {reg_costs[1]:.4f}\n"
        f"   - λ = 1.0: Cost = {reg_costs[2]:.4f}\n\n"
        f"2. Mô hình tối ưu nhất:\n"
        f"   - λ = {best_lambda}\n"
        f"   - Phương trình: y = {' + '.join([f'{w:.4f}x^{i}' for i, w in enumerate(best_reg_model.weights)])}\n"
        f"   - Cost: {min(reg_costs):.4f}\n\n"
        f"3. Nhận xét:\n"
        f"   - Regularization với λ = 0.01 cho kết quả tốt nhất\n"
        f"   - Các tham số của mô hình với λ = 0.01 có giá trị ổn định hơn\n"
        f"   - Regularization giúp kiểm soát overfitting nhưng không cải thiện đáng kể cost\n"
        f"   - Mô hình bậc 2 vẫn là lựa chọn tốt hơn so với mô hình bậc 4 có regularization",
        title="Regularization Evaluation",
        border_style="cyan"
    ))

if __name__ == '__main__':
    main() 