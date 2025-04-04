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
from .core.cost_function import (
    compute_cost, compute_cost_2d,
    generate_house_data, generate_house_data_2d
)
from .core.gradient_descent import (
    gradient_descent, gradient_descent_2d,
    compute_gradient, compute_gradient_2d
)

# Import visualization
from .visualization.cost_plot import (
    plot_cost_3d,
    plot_cost_contour,
    plot_linear_regression_fit
)
from .visualization.gradient_plot import plot_gradient_descent, plot_gradient_steps

# Import logistic regression
from .core.logistic_regression import (
    generate_admission_data,
    gradient_descent_logistic,
    compute_cost_logistic
)
from .visualization.logistic_plot import (
    plot_decision_boundary,
    plot_cost_surface_logistic,
    plot_gradient_descent_logistic
)

# Initialize rich console
console = Console()

# Dictionary quản lý hình ảnh
IMAGES = {
    "Cost Function": {
        "cost_function_3d.png": "Bề mặt cost function trong không gian 3D",
        "cost_function_contour.png": "Đường đồng mức của cost function",
        "logistic_cost_3d.png": "Bề mặt cost function cho logistic regression"
    },
    "Gradient Descent": {
        "gradient_descent_3d.png": "Quá trình gradient descent trên bề mặt cost 3D",
        "gradient_descent_contour.png": "Quá trình gradient descent trên contour",
        "gradient_descent_steps.png": "Các bước của gradient descent trên dữ liệu",
        "cost_history.png": "Lịch sử cost function qua các iteration",
        "decision_boundary.png": "Decision boundary cho logistic regression",
        "logistic_gradient_descent.png": "Quá trình gradient descent cho logistic regression"
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
    """Chạy các ví dụ về vector cơ bản"""
    console.print("\n[bold cyan]1. Vector Operations[/bold cyan]", justify="center")
    
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

def run_cost_and_gradient_example():
    """Chạy ví dụ về cost function và gradient descent"""
    console.print("\n[bold cyan]2. Cost Function và Gradient Descent[/bold cyan]", justify="center")
    
    # Phần 1: Ví dụ với 1 tham số
    console.print("\n[bold magenta]2.1. Ví dụ với 1 tham số (kích thước nhà)[/bold magenta]")
    run_one_feature_example()
    
    # Phần 2: Ví dụ với 2 tham số
    console.print("\n[bold magenta]2.2. Ví dụ với 2 tham số (kích thước và số phòng ngủ)[/bold magenta]")
    run_two_features_example()

def run_one_feature_example():
    """Chạy ví dụ với 1 tham số"""
    # Tạo dữ liệu mẫu về giá nhà
    x_train, y_train = generate_house_data()
    
    # Hiển thị thông tin về ví dụ
    console.print(Panel(
        "[bold green]Thông tin về ví dụ:[/bold green]\n"
        "1. Dữ liệu mẫu:\n"
        "   - x: kích thước nhà (1000 sqft)\n"
        "   - y: giá nhà (1000s $)\n"
        "   - Số mẫu: {len(x_train)}\n\n"
        "2. Công thức:\n"
        "   - Hàm dự đoán: f(x) = wx + b\n"
        "   - Cost function: J(w,b) = (1/2m) * Σ(f(x⁽ⁱ⁾) - y⁽ⁱ⁾)²\n"
        "   - Gradient:\n"
        "     * ∂J/∂w = (1/m) * Σ(f(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x⁽ⁱ⁾\n"
        "     * ∂J/∂b = (1/m) * Σ(f(x⁽ⁱ⁾) - y⁽ⁱ⁾)\n"
        "   - Cập nhật tham số:\n"
        "     * w = w - α * ∂J/∂w\n"
        "     * b = b - α * ∂J/∂b",
        title="Example Overview",
        border_style="cyan"
    ))
    
    # Hiển thị dữ liệu mẫu
    console.print(Panel(
        f"[bold green]Dữ liệu training:[/bold green]\n"
        f"Kích thước nhà: {x_train}\n"
        f"Giá nhà: {y_train}",
        title="Training Data",
        border_style="cyan"
    ))
    
    # Tính cost function tại một điểm
    w_example, b_example = 180, 150  # Thay đổi điểm minh họa
    cost = compute_cost(x_train, y_train, w_example, b_example)
    
    # Debug prints
    console.print(f"\n[bold red]Debug:[/bold red]")
    console.print(f"x_train: {x_train}")
    console.print(f"y_train: {y_train}")
    console.print(f"w_example: {w_example}")
    console.print(f"b_example: {b_example}")
    console.print(f"Actual cost: {cost}")
    
    console.print(Panel(
        f"[bold green]Minh họa Cost Function tại một điểm:[/bold green]\n"
        f"Chọn điểm (w={w_example}, b={b_example}) để minh họa cách tính cost function:\n"
        f"- w = {w_example}: giá tăng {w_example}$ cho mỗi 1000 sqft\n"
        f"- b = {b_example}: giá cơ bản {b_example}$1000\n"
        f"- Cost = {cost:.2f}: độ lệch trung bình bình phương của dự đoán\n\n"
        f"[yellow]Giải thích:[/yellow]\n"
        f"Tại điểm này, mô hình có một số sai số trong dự đoán:\n"
        f"- Nhà 1000 sqft: dự đoán = {w_example*1 + b_example} = {w_example*1 + b_example:.0f} (thực tế: 300)\n"
        f"- Nhà 2000 sqft: dự đoán = {w_example*2 + b_example} = {w_example*2 + b_example:.0f} (thực tế: 500)\n"
        f"- Nhà 3000 sqft: dự đoán = {w_example*3 + b_example} = {w_example*3 + b_example:.0f} (thực tế: 700)\n"
        f"- Nhà 4000 sqft: dự đoán = {w_example*4 + b_example} = {w_example*4 + b_example:.0f} (thực tế: 900)\n"
        f"- Nhà 5000 sqft: dự đoán = {w_example*5 + b_example} = {w_example*5 + b_example:.0f} (thực tế: 1100)\n\n"
        f"[yellow]Lưu ý:[/yellow] Đây là một điểm bất kỳ trên bề mặt cost function.\n"
        f"Trong phần tiếp theo, chúng ta sẽ tìm điểm tối ưu (w*, b*) có cost thấp nhất.",
        title="Cost Function Evaluation",
        border_style="cyan"
    ))
    
    # Vẽ đồ thị cost function
    with console.status("[bold green]Tạo đồ thị cost function..."):
        plot_cost_3d(x_train, y_train, w_range=(100, 300), b_range=(-200, 200), save_as='cost_function_3d.png')
        plot_cost_contour(x_train, y_train, w_range=(100, 300), b_range=(-200, 200), save_as='cost_function_contour.png')
    
    # Thực hiện gradient descent
    initial_w, initial_b = 100, 0
    iterations = 1000
    alpha = 0.01
    
    console.print(Panel(
        f"[bold green]Thông tin gradient descent:[/bold green]\n"
        f"- Learning rate (α): {alpha}\n"
        f"- Số iteration: {iterations}\n"
        f"- Tham số khởi tạo: w = {initial_w}, b = {initial_b}",
        title="Gradient Descent Setup",
        border_style="cyan"
    ))
    
    with console.status("[bold green]Running gradient descent..."):
        w_final, b_final, J_hist, p_hist = gradient_descent(
            x_train, y_train, initial_w, initial_b, alpha, iterations, compute_cost)
    
    # Tách lịch sử tham số
    w_hist = [p[0] for p in p_hist]
    b_hist = [p[1] for p in p_hist]
    
    # Hiển thị kết quả
    console.print(Panel(
        f"[bold green]Kết quả tối ưu sau {iterations} iterations:[/bold green]\n"
        f"1. Tham số tối ưu (w*, b*):\n"
        f"   w* = {w_final:.2f}\n"
        f"   b* = {b_final:.2f}\n"
        f"2. Phương trình hồi quy tối ưu:\n"
        f"   y = {w_final:.2f}x + {b_final:.2f}\n"
        f"3. Cost tối ưu: {J_hist[-1]:.4f}\n\n"
        f"[yellow]So sánh với điểm minh họa:[/yellow]\n"
        f"- Điểm minh họa (w={w_example}, b={b_example}): Cost = {cost:.2f}\n"
        f"- Điểm tối ưu (w*={w_final:.2f}, b*={b_final:.2f}): Cost = {J_hist[-1]:.4f}\n"
        f"→ Điểm tối ưu có cost thấp hơn, nghĩa là mô hình dự đoán chính xác hơn.",
        title="Optimization Results",
        border_style="cyan"
    ))
    
    # Tạo bảng theo dõi cost
    cost_table = Table(title="Theo dõi Cost qua các Iteration")
    cost_table.add_column("Iteration", style="cyan", justify="right")
    cost_table.add_column("Cost", style="green", justify="right")
    cost_table.add_column("Thay đổi", style="yellow", justify="right")
    
    # Thêm các mốc quan trọng
    milestones = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, iterations-1]
    for i in milestones:
        if i < len(J_hist):
            cost = J_hist[i]
            if i > 0:
                change = J_hist[i] - J_hist[i-1]
                change_str = f"{change:+.4f}"
            else:
                change_str = "-"
            cost_table.add_row(
                f"{i:4d}",
                f"{cost:.4f}",
                change_str
            )
    
    console.print(cost_table)
    
    # Vẽ đồ thị gradient descent
    with console.status("[bold green]Tạo đồ thị gradient descent..."):
        plot_gradient_descent(x_train, y_train, w_hist, b_hist, J_hist, compute_cost, save_as='gradient_descent_3d.png')
        plot_gradient_steps(x_train, y_train, w_hist, b_hist, compute_cost, save_as='gradient_descent_steps.png')

def run_two_features_example():
    """Chạy ví dụ với 2 tham số"""
    # Tạo dữ liệu mẫu về giá nhà
    x1_train, x2_train, y_train = generate_house_data_2d()
    
    # Hiển thị thông tin về ví dụ
    console.print(Panel(
        "[bold green]Thông tin về ví dụ 2 tham số:[/bold green]\n"
        "1. Dữ liệu mẫu:\n"
        "   - x₁: kích thước nhà (1000 sqft)\n"
        "   - x₂: số phòng ngủ\n"
        "   - y: giá nhà (1000s $)\n"
        "   - Số mẫu: {len(x1_train)}\n\n"
        "2. Công thức:\n"
        "   - Hàm dự đoán: f(x₁,x₂) = w₁x₁ + w₂x₂ + b\n"
        "   - Cost function: J(w₁,w₂,b) = (1/2m) * Σ(f(x₁⁽ⁱ⁾,x₂⁽ⁱ⁾) - y⁽ⁱ⁾)²\n"
        "   - Gradient:\n"
        "     * ∂J/∂w₁ = (1/m) * Σ(f(x₁⁽ⁱ⁾,x₂⁽ⁱ⁾) - y⁽ⁱ⁾) * x₁⁽ⁱ⁾\n"
        "     * ∂J/∂w₂ = (1/m) * Σ(f(x₁⁽ⁱ⁾,x₂⁽ⁱ⁾) - y⁽ⁱ⁾) * x₂⁽ⁱ⁾\n"
        "     * ∂J/∂b = (1/m) * Σ(f(x₁⁽ⁱ⁾,x₂⁽ⁱ⁾) - y⁽ⁱ⁾)",
        title="Example Overview (2 Features)",
        border_style="cyan"
    ))
    
    # Hiển thị dữ liệu mẫu
    console.print(Panel(
        f"[bold green]Dữ liệu training:[/bold green]\n"
        f"Kích thước nhà (x₁): {x1_train}\n"
        f"Số phòng ngủ (x₂): {x2_train}\n"
        f"Giá nhà (y): {y_train}",
        title="Training Data (2 Features)",
        border_style="cyan"
    ))
    
    # Tính cost function tại một điểm
    w1_example, w2_example, b_example = 150, 50, 100
    cost = compute_cost_2d(x1_train, x2_train, y_train, w1_example, w2_example, b_example)
    
    console.print(Panel(
        f"[bold green]Minh họa Cost Function tại một điểm:[/bold green]\n"
        f"Chọn điểm (w₁={w1_example}, w₂={w2_example}, b={b_example}) để minh họa:\n"
        f"- w₁ = {w1_example}: giá tăng {w1_example}$ cho mỗi 1000 sqft\n"
        f"- w₂ = {w2_example}: giá tăng {w2_example}$ cho mỗi phòng ngủ\n"
        f"- b = {b_example}: giá cơ bản {b_example}$1000\n"
        f"- Cost = {cost:.2f}: độ lệch trung bình bình phương của dự đoán\n\n"
        f"[yellow]Giải thích:[/yellow]\n"
        f"Tại điểm này, mô hình dự đoán:\n"
        f"- Nhà 1000sqft, 1 phòng ngủ: {w1_example*1 + w2_example*1 + b_example:.0f} (thực tế: 250)\n"
        f"- Nhà 2000sqft, 1 phòng ngủ: {w1_example*2 + w2_example*1 + b_example:.0f} (thực tế: 350)\n"
        f"- Nhà 2000sqft, 2 phòng ngủ: {w1_example*2 + w2_example*2 + b_example:.0f} (thực tế: 400)\n"
        f"- Nhà 3000sqft, 2 phòng ngủ: {w1_example*3 + w2_example*2 + b_example:.0f} (thực tế: 500)\n"
        f"- Nhà 3000sqft, 3 phòng ngủ: {w1_example*3 + w2_example*3 + b_example:.0f} (thực tế: 550)\n"
        f"- Nhà 4000sqft, 3 phòng ngủ: {w1_example*4 + w2_example*3 + b_example:.0f} (thực tế: 650)",
        title="Cost Function Evaluation (2 Features)",
        border_style="cyan"
    ))
    
    # Thực hiện gradient descent
    initial_w1, initial_w2, initial_b = 100, 0, 0
    iterations = 1000
    alpha = 0.01
    
    console.print(Panel(
        f"[bold green]Thông tin gradient descent:[/bold green]\n"
        f"- Learning rate (α): {alpha}\n"
        f"- Số iteration: {iterations}\n"
        f"- Tham số khởi tạo:\n"
        f"  * w₁ = {initial_w1}\n"
        f"  * w₂ = {initial_w2}\n"
        f"  * b = {initial_b}",
        title="Gradient Descent Setup (2 Features)",
        border_style="cyan"
    ))
    
    with console.status("[bold green]Running gradient descent..."):
        w1_final, w2_final, b_final, J_hist, p_hist = gradient_descent_2d(
            x1_train, x2_train, y_train,
            initial_w1, initial_w2, initial_b,
            alpha, iterations, compute_cost_2d)
    
    # Hiển thị kết quả
    console.print(Panel(
        f"[bold green]Kết quả tối ưu sau {iterations} iterations:[/bold green]\n"
        f"1. Tham số tối ưu (w₁*, w₂*, b*):\n"
        f"   w₁* = {w1_final:.2f}: ảnh hưởng của kích thước nhà\n"
        f"   w₂* = {w2_final:.2f}: ảnh hưởng của số phòng ngủ\n"
        f"   b* = {b_final:.2f}: giá cơ bản\n"
        f"2. Phương trình hồi quy tối ưu:\n"
        f"   y = {w1_final:.2f}x₁ + {w2_final:.2f}x₂ + {b_final:.2f}\n"
        f"3. Cost tối ưu: {J_hist[-1]:.4f}\n\n"
        f"[yellow]So sánh với điểm minh họa:[/yellow]\n"
        f"- Điểm minh họa (w₁={w1_example}, w₂={w2_example}, b={b_example}): Cost = {cost:.2f}\n"
        f"- Điểm tối ưu (w₁*={w1_final:.2f}, w₂*={w2_final:.2f}, b*={b_final:.2f}): Cost = {J_hist[-1]:.4f}\n"
        f"→ Điểm tối ưu có cost thấp hơn, nghĩa là mô hình dự đoán chính xác hơn.",
        title="Optimization Results (2 Features)",
        border_style="cyan"
    ))
    
    # Tạo bảng theo dõi cost
    cost_table = Table(title="Theo dõi Cost qua các Iteration (2 Features)")
    cost_table.add_column("Iteration", style="cyan", justify="right")
    cost_table.add_column("Cost", style="green", justify="right")
    cost_table.add_column("Thay đổi", style="yellow", justify="right")
    
    # Thêm các mốc quan trọng
    milestones = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, iterations-1]
    for i in milestones:
        if i < len(J_hist):
            cost = J_hist[i]
            if i > 0:
                change = J_hist[i] - J_hist[i-1]
                change_str = f"{change:+.4f}"
            else:
                change_str = "-"
            cost_table.add_row(
                f"{i:4d}",
                f"{cost:.4f}",
                change_str
            )
    
    console.print(cost_table)

def run_logistic_regression_example():
    """Chạy ví dụ về logistic regression"""
    console.print("\n[bold cyan]3. Logistic Regression Example[/bold cyan]", justify="center")
    
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
    
    # Tạo dữ liệu mẫu
    x1, x2, y = generate_admission_data()
    
    # Hiển thị thông tin về dữ liệu
    console.print(Panel(
        f"[bold green]Dữ liệu tuyển sinh:[/bold green]\n"
        f"- Số lượng mẫu: {len(y)}\n"
        f"- Số lượng sinh viên đỗ: {np.sum(y)}\n"
        f"- Số lượng sinh viên trượt: {len(y) - np.sum(y)}\n"
        f"- Điểm thi: min={x1.min():.1f}, max={x1.max():.1f}, mean={x1.mean():.1f}\n"
        f"- GPA: min={x2.min():.1f}, max={x2.max():.1f}, mean={x2.mean():.1f}",
        title="Training Data",
        border_style="cyan"
    ))
    
    # Thực hiện gradient descent
    initial_w1, initial_w2, initial_b = 0, 0, 0
    iterations = 1000
    alpha = 0.01
    
    console.print(Panel(
        f"[bold green]Thông tin gradient descent:[/bold green]\n"
        f"- Learning rate (α): {alpha}\n"
        f"- Số iteration: {iterations}\n"
        f"- Tham số khởi tạo:\n"
        f"  * w₁ = {initial_w1}\n"
        f"  * w₂ = {initial_w2}\n"
        f"  * b = {initial_b}",
        title="Gradient Descent Setup",
        border_style="cyan"
    ))
    
    with console.status("[bold green]Running gradient descent..."):
        w1_final, w2_final, b_final, J_hist, p_hist = gradient_descent_logistic(
            x1, x2, y, initial_w1, initial_w2, initial_b, alpha, iterations)
    
    # Tách lịch sử tham số
    w1_hist = [p[0] for p in p_hist]
    w2_hist = [p[1] for p in p_hist]
    b_hist = [p[2] for p in p_hist]
    
    # Hiển thị kết quả
    console.print(Panel(
        f"[bold green]Kết quả tối ưu sau {iterations} iterations:[/bold green]\n"
        f"1. Tham số tối ưu (w₁*, w₂*, b*):\n"
        f"   w₁* = {w1_final:.4f}: trọng số điểm thi\n"
        f"   w₂* = {w2_final:.4f}: trọng số GPA\n"
        f"   b* = {b_final:.4f}: độ chệch\n"
        f"2. Phương trình phân loại:\n"
        f"   P(đỗ) = g({w1_final:.4f}x₁ + {w2_final:.4f}x₂ + {b_final:.4f})\n"
        f"3. Cost tối ưu: {J_hist[-1]:.4f}",
        title="Optimization Results",
        border_style="cyan"
    ))
    
    # Tạo bảng theo dõi cost
    cost_table = Table(title="Theo dõi Cost qua các Iteration")
    cost_table.add_column("Iteration", style="cyan", justify="right")
    cost_table.add_column("Cost", style="green", justify="right")
    cost_table.add_column("Thay đổi", style="yellow", justify="right")
    
    # Thêm các mốc quan trọng
    milestones = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, iterations-1]
    for i in milestones:
        if i < len(J_hist):
            cost = J_hist[i]
            if i > 0:
                change = J_hist[i] - J_hist[i-1]
                change_str = f"{change:+.4f}"
            else:
                change_str = "-"
            cost_table.add_row(
                f"{i:4d}",
                f"{cost:.4f}",
                change_str
            )
    
    console.print(cost_table)
    
    # Vẽ các đồ thị
    with console.status("[bold green]Tạo đồ thị..."):
        plot_decision_boundary(x1, x2, y, w1_final, w2_final, b_final)
        plot_cost_surface_logistic(x1, x2, y, [-1, 1], [-1, 1], b_final)
        plot_gradient_descent_logistic(x1, x2, y, w1_hist, w2_hist, b_hist, J_hist)

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
        run_cost_and_gradient_example,
        run_logistic_regression_example
    ], description="Running examples..."):
        example()
    
    # In thông tin về các hình ảnh đã tạo
    print_generated_images()

if __name__ == "__main__":
    main() 