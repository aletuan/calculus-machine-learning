"""
Main script to run all examples.
"""

import os
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track
from rich.tree import Tree
from rich import print as rprint
import click

# Import core functionality
from .core.linear.regression import LinearRegression
from .core.logistic.regression import LogisticRegression

# Import examples
from .examples.linear_example import run_linear_example
from .examples.linear_example_multiple import run_multiple_example
from .examples.logistic_example import run_logistic_example
from .examples.polynomial_example import main as polynomial_main

# Initialize rich console
console = Console()

# Dictionary quản lý hình ảnh
IMAGES = {
    "Linear Regression": {
        "linear_regression_fit.png": "Đường hồi quy tuyến tính và dữ liệu",
        "linear_cost_history.png": "Lịch sử cost function của linear regression",
        "multiple_regression_fit.png": "Mặt phẳng hồi quy nhiều biến và dữ liệu",
        "multiple_cost_history.png": "Lịch sử cost function của hồi quy nhiều biến"
    },
    "Polynomial Regression": {
        "polynomial_regression_fit.png": "So sánh các mô hình polynomial khác bậc",
        "regularization_effect.png": "Ảnh hưởng của regularization"
    },
    "Logistic Regression": {
        "logistic_decision_boundary.png": "Decision boundary của logistic regression",
        "logistic_cost_history.png": "Lịch sử cost function của logistic regression"
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

@click.command()
@click.option('--example', type=click.Choice(['linear', 'multiple', 'polynomial', 'logistic', 'all']), 
              default='all', help='Chọn ví dụ để chạy')
def main(example):
    """Chạy các ví dụ về machine learning"""
    ensure_images_dir()
    
    if example in ['linear', 'all']:
        console.print(Panel(
            "[bold cyan]Ví dụ 1: Hồi quy tuyến tính đơn giản[/bold cyan]\n"
            "Dự đoán giá nhà dựa trên diện tích",
            border_style="cyan"
        ))
        run_linear_example()
    
    if example in ['multiple', 'all']:
        console.print(Panel(
            "[bold cyan]Ví dụ 2: Hồi quy tuyến tính nhiều biến[/bold cyan]\n"
            "Dự đoán giá nhà dựa trên diện tích và số phòng ngủ",
            border_style="cyan"
        ))
        run_multiple_example()

    if example in ['polynomial', 'all']:
        console.print(Panel(
            "[bold cyan]Ví dụ 3: Polynomial Regression với Regularization[/bold cyan]\n"
            "Dự đoán giá nhà với mô hình phi tuyến và kiểm soát overfitting",
            border_style="cyan"
        ))
        polynomial_main()
    
    if example in ['logistic', 'all']:
        console.print(Panel(
            "[bold cyan]Ví dụ 4: Hồi quy logistic[/bold cyan]\n"
            "Dự đoán kết quả tuyển sinh dựa trên điểm thi và GPA",
            border_style="cyan"
        ))
        run_logistic_example()

    # In thông tin về các hình ảnh đã tạo
    print_generated_images()

if __name__ == "__main__":
    main() 