import os
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track

# Import core functionality
from .core.vector import (
    add_vectors,
    dot_product,
    vector_norm
)
from .core.linear_model import find_linear_regression
from .core.cost_function import compute_cost, generate_house_data
from .core.gradient_descent import gradient_descent

# Import visualization
from .visualization.cost_plot import (
    plot_cost_surface,
    plot_cost_contour
)
from .visualization.gradient_plot import plot_gradient_descent, plot_gradient_steps

# Initialize rich console
console = Console()

def run_vector_examples():
    """Chạy các ví dụ về vector"""
    console.print("\n[bold cyan]Vector Operations Examples[/bold cyan]", justify="center")
    
    # Create table for vector operations
    table = Table(title="Vector Operations Results")
    table.add_column("Operation", style="cyan")
    table.add_column("Result", style="green")
    
    v1 = [3, 4]
    v2 = [1, 2]
    
    table.add_row("Vector 1", str(v1))
    table.add_row("Vector 2", str(v2))
    table.add_row("Sum", str(add_vectors(v1, v2)))
    table.add_row("Dot product", str(dot_product(v1, v2)))
    table.add_row("Norm of v1", f"{vector_norm(v1):.2f}")
    
    console.print(table)

def run_linear_regression():
    """Chạy ví dụ về linear regression"""
    console.print("\n[bold cyan]Linear Regression Example[/bold cyan]", justify="center")
    
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    w, b = find_linear_regression(x, y)
    
    panel = Panel(
        f"[green]Found parameters:[/green]\n"
        f"w = {w:.2f}\n"
        f"b = {b:.2f}\n"
        f"Equation: y = {w:.2f}x + {b:.2f}",
        title="Linear Regression Results",
        border_style="cyan"
    )
    console.print(panel)

def run_cost_visualization():
    """Chạy ví dụ về cost function visualization"""
    console.print("\n[bold cyan]Cost Function Visualization[/bold cyan]", justify="center")
    
    with console.status("[bold green]Generating cost function visualizations..."):
        x_train, y_train = generate_house_data()
        w, b = 200, 100
        cost = compute_cost(x_train, y_train, w, b)
        
        panel = Panel(
            f"[green]Cost function evaluation:[/green]\n"
            f"w = {w}\n"
            f"b = {b}\n"
            f"Cost = {cost:.2f}",
            title="Cost Function Results",
            border_style="cyan"
        )
        console.print(panel)
        
        plot_cost_surface(x_train, y_train)
        plot_cost_contour(x_train, y_train)
        
        console.print("[green]✓[/green] Cost function visualizations saved to 'images' directory")

def run_gradient_descent():
    """Chạy ví dụ về gradient descent"""
    console.print("\n[bold cyan]Gradient Descent Example[/bold cyan]", justify="center")
    
    x_train, y_train = generate_house_data()
    initial_w = 100
    initial_b = 0
    iterations = 1000
    alpha = 0.01
    
    console.print(Panel(
        f"[green]Initial parameters:[/green]\n"
        f"w = {initial_w}\n"
        f"b = {initial_b}\n"
        f"Learning rate (α) = {alpha}\n"
        f"Iterations = {iterations}",
        title="Gradient Descent Setup",
        border_style="cyan"
    ))
    
    # Thực hiện gradient descent
    with console.status("[bold green]Running gradient descent..."):
        w_final, b_final, J_hist, p_hist = gradient_descent(
            x_train, y_train, initial_w, initial_b, alpha, iterations, compute_cost)
    
    # Tách lịch sử tham số
    w_hist = [p[0] for p in p_hist]
    b_hist = [p[1] for p in p_hist]
    
    console.print(Panel(
        f"[green]Final results:[/green]\n"
        f"w = {w_final:.2f}\n"
        f"b = {b_final:.2f}\n"
        f"Final cost = {J_hist[-1]:.2f}",
        title="Gradient Descent Results",
        border_style="cyan"
    ))
    
    # Vẽ đồ thị
    with console.status("[bold green]Generating visualizations..."):
        plot_gradient_descent(x_train, y_train, w_hist, b_hist, J_hist, compute_cost)
        plot_gradient_steps(x_train, y_train, w_hist, b_hist, compute_cost)
        console.print("[green]✓[/green] Gradient descent visualizations saved to 'images' directory")

def main():
    """Hàm main chạy tất cả các ví dụ"""
    console.print("[bold magenta]Calculus Machine Learning Examples[/bold magenta]", justify="center")
    console.print("=" * 50, justify="center")
    
    # Ensure images directory exists
    if not os.path.exists('images'):
        os.makedirs('images')
    
    # Chạy các ví dụ
    for example in track([
        run_vector_examples,
        run_linear_regression,
        run_cost_visualization,
        run_gradient_descent
    ], description="Running examples..."):
        example()
    
    console.print("\n[bold green]✓ All examples completed![/bold green]")
    console.print("[dim]Check 'images' directory for visualizations.[/dim]")

if __name__ == "__main__":
    main() 