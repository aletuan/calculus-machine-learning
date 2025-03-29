import os
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track
from rich.tree import Tree
from rich import print as rprint

# Import core functionality
from .core.vector import (
    add_vectors,
    dot_product,
    vector_norm,
    create_vectors,
    vector_operations,
    unit_vector_and_angle,
    vector_projection
)
from .core.cost_function import compute_cost, generate_house_data
from .core.gradient_descent import gradient_descent, compute_gradient

# Import visualization
from .visualization.cost_plot import (
    plot_cost_3d,
    plot_cost_contour,
    plot_linear_regression_fit
)
from .visualization.gradient_plot import plot_gradient_descent, plot_gradient_steps

# Initialize rich console
console = Console()

# Dictionary quản lý hình ảnh
IMAGES = {
    "Linear Regression": {
        "linear_regression_fit.png": "Kết quả hồi quy tuyến tính với dữ liệu training và đường hồi quy"
    },
    "Cost Function": {
        "cost_function_3d.png": "Bề mặt cost function trong không gian 3D",
        "cost_function_contour.png": "Đường đồng mức của cost function"
    },
    "Gradient Descent": {
        "gradient_descent_3d.png": "Quá trình gradient descent trên bề mặt cost 3D",
        "gradient_descent_contour.png": "Quá trình gradient descent trên contour",
        "gradient_descent_steps.png": "Các bước của gradient descent trên dữ liệu",
        "cost_history.png": "Lịch sử cost function qua các iteration"
    }
}

def ensure_images_dir():
    """Đảm bảo thư mục images tồn tại"""
    if not os.path.exists('images'):
        os.makedirs('images')

def print_generated_images():
    """In thông tin về các hình ảnh đã tạo"""
    tree = Tree("📊 Hình ảnh đã tạo")
    
    for category, images in IMAGES.items():
        category_tree = tree.add(f"📁 {category}")
        for img_name, description in images.items():
            img_path = os.path.join('images', img_name)
            if os.path.exists(img_path):
                size = os.path.getsize(img_path) / 1024  # Convert to KB
                category_tree.add(f"📄 {img_name} ({size:.1f}KB) - {description}")
            else:
                category_tree.add(f"❌ {img_name} (không tìm thấy) - {description}")
    
    console.print("\n")
    console.print(Panel(tree, title="[bold blue]Thông tin hình ảnh[/bold blue]"))
    console.print("\n")

def run_vector_examples():
    """Chạy các ví dụ về vector"""
    console.print("\n[bold cyan]Vector Operations Examples[/bold cyan]", justify="center")
    
    # Create sample vectors
    v1, v2 = create_vectors()
    
    # Perform vector operations
    operations = vector_operations(v1, v2)
    unit_angle = unit_vector_and_angle(v1, v2)
    projection = vector_projection(v1, v2)
    
    # Create table for results
    table = Table(title="Vector Operations Results")
    table.add_column("Operation", style="cyan")
    table.add_column("Result", style="green")
    
    table.add_row("Vector 1", str(v1))
    table.add_row("Vector 2", str(v2))
    table.add_row("Sum", str(operations['v_sum']))
    table.add_row("Difference", str(operations['v_diff']))
    table.add_row("Scaled (2*v1)", str(operations['v_scaled']))
    table.add_row("Norm of v1", f"{operations['v1_norm']:.2f}")
    table.add_row("Dot product", str(operations['dot_product']))
    table.add_row("Cross product", str(operations['cross_product']))
    table.add_row("Angle between vectors", f"{unit_angle['angle']:.2f}°")
    table.add_row("Projection norm", f"{projection['projection_norm']:.2f}")
    
    console.print(table)

def run_linear_regression():
    """Chạy ví dụ về linear regression sử dụng gradient descent"""
    console.print("\n[bold cyan]Linear Regression Example[/bold cyan]", justify="center")
    
    # Tạo dữ liệu mẫu
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    
    # Khởi tạo tham số
    initial_w = 0
    initial_b = 0
    iterations = 1000
    alpha = 0.01
    
    # Thực hiện gradient descent
    with console.status("[bold green]Running gradient descent..."):
        w_final, b_final, J_hist, p_hist = gradient_descent(
            x, y, initial_w, initial_b, alpha, iterations, compute_cost)
    
    panel = Panel(
        f"[green]Found parameters:[/green]\n"
        f"w = {w_final:.2f}\n"
        f"b = {b_final:.2f}\n"
        f"Equation: y = {w_final:.2f}x + {b_final:.2f}",
        title="Linear Regression Results",
        border_style="cyan"
    )
    console.print(panel)
    
    # Create and save visualization
    with console.status("[bold green]Generating visualization..."):
        plot_linear_regression_fit(x, y, w_final, b_final, save_as='linear_regression_fit.png')
        console.print("[green]✓[/green] Linear regression visualization saved")

def run_cost_calculation():
    """Chạy ví dụ về cost calculation"""
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
        
        plot_cost_3d(x_train, y_train, w_range=(100, 300), b_range=(-200, 200), save_as='cost_function_3d.png')
        plot_cost_contour(x_train, y_train, w_range=(100, 300), b_range=(-200, 200), save_as='cost_function_contour.png')
        
        console.print("[green]✓[/green] Cost function visualizations saved")

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
        plot_gradient_descent(x_train, y_train, w_hist, b_hist, J_hist, compute_cost, save_as='gradient_descent_3d.png')
        plot_gradient_steps(x_train, y_train, w_hist, b_hist, compute_cost, save_as='gradient_descent_steps.png')
        console.print("[green]✓[/green] Gradient descent visualizations saved")

def main():
    """Hàm main chạy tất cả các ví dụ"""
    console.print(Panel.fit(
        "[bold blue]Ứng Dụng Giải Tích và Học Máy[/bold blue]\n"
        "[italic]Minh họa các khái niệm cơ bản trong học máy[/italic]",
        border_style="blue"
    ))
    
    # Ensure images directory exists
    ensure_images_dir()
    
    # Chạy các ví dụ
    for example in track([
        run_vector_examples,
        run_linear_regression,
        run_cost_calculation,
        run_gradient_descent
    ], description="Running examples..."):
        example()
    
    # In thông tin về các hình ảnh đã tạo
    print_generated_images()

if __name__ == "__main__":
    main() 