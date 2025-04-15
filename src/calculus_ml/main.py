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
from .examples.linear_regression.single_linear_example import run_linear_example
from .examples.linear_regression.multiple_linear_example import run_multiple_example
from .examples.logistic_example import run_logistic_example
from .examples.polynomial_example import main as polynomial_main
from .examples.perceptron.train import train_perceptron
from .examples.single_hidden_layer.train import main as run_neural_network
from .examples.tf_one_hidden_layer.train import main as run_tf_neural_network

# Initialize rich console
console = Console()

# Dictionary quản lý hình ảnh
IMAGES = {
    "Linear Regression": {
        "linear_regression_fit.png": "Đường hồi quy tuyến tính và dữ liệu",
        "linear_cost_history.png": "Lịch sử cost function của linear regression",
        "multiple_regression_fit.png": "Mặt phẳng hồi quy nhiều biến và dữ liệu",
        "multiple_cost_history.png": "Lịch sử cost function của hồi quy nhiều biến",
        "gradient_descent_example.png": "Minh họa quá trình gradient descent"
    },
    "Polynomial Regression": {
        "polynomial_regression_fit.png": "So sánh các mô hình polynomial khác bậc",
        "regularization_effect.png": "Ảnh hưởng của regularization"
    },
    "Logistic Regression": {
        "logistic_decision_boundary.png": "Decision boundary của logistic regression",
        "logistic_cost_history.png": "Lịch sử cost function của logistic regression"
    },
    "Perceptron": {
        "perceptron_training_history.png": "Lịch sử training của perceptron"
    },
    "Neural Network": {
        "neural_network_training_history.png": "Lịch sử training của neural network"
    },
    "TensorFlow Neural Network": {
        "tf_and_training.png": "Lịch sử training của TensorFlow neural network"
    },
    "Decision Tree": {
        "decision_tree_boundary.png": "Decision boundary của Decision Tree"
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

def print_examples_table():
    """In bảng các ví dụ có sẵn"""
    table = Table(title="Machine Learning Examples")
    table.add_column("Example", style="cyan", justify="left")
    table.add_column("Description", style="green", justify="left")
    
    examples = [
        ("1. Linear Regression", 
         "Dự đoán giá nhà dựa trên diện tích"),
        ("2. Multiple Linear Regression",
         "Dự đoán giá nhà dựa trên diện tích và số phòng ngủ"),
        ("3. Polynomial Regression",
         "Dự đoán giá nhà với các mô hình đa thức"),
        ("4. Logistic Regression",
         "Phân loại học sinh đỗ/trượt dựa trên điểm thi"),
        ("5. Perceptron",
         "Học hàm AND với perceptron"),
        ("6. Neural Network",
         "Học hàm XOR với neural network một lớp ẩn"),
        ("7. TensorFlow Neural Network",
         "Học hàm AND với neural network sử dụng TensorFlow"),
        ("8. Decision Tree",
         "Phân loại hoa Iris sử dụng Decision Tree")
    ]
    
    for example in examples:
        table.add_row(*example)
    
    console.print("\n")
    console.print(Panel(table, title="[bold blue]Available Examples[/bold blue]"))
    console.print("\n")

def get_user_choice():
    """Lấy lựa chọn của người dùng"""
    while True:
        try:
            console.print("\n[bold]Chọn ví dụ để chạy:[/bold]")
            console.print("1. Chạy tất cả các ví dụ")
            console.print("2. Chọn ví dụ cụ thể")
            choice = int(console.input("\n[bold]Nhập lựa chọn (1-2): [/bold]"))
            if choice in [1, 2]:
                break
            console.print("[red]Vui lòng nhập 1 hoặc 2[/red]")
        except ValueError:
            console.print("[red]Vui lòng nhập số hợp lệ[/red]")
    
    if choice == 1:
        return 'all'
    
    while True:
        try:
            console.print("\n[bold]Chọn ví dụ cụ thể:[/bold]")
            console.print("1. Linear Regression")
            console.print("2. Multiple Linear Regression")
            console.print("3. Polynomial Regression")
            console.print("4. Logistic Regression")
            console.print("5. Perceptron")
            console.print("6. Neural Network")
            console.print("7. TensorFlow Neural Network")
            console.print("8. Decision Tree")
            example_choice = int(console.input("\n[bold]Nhập số ví dụ (1-8): [/bold]"))
            if 1 <= example_choice <= 8:
                break
            console.print("[red]Vui lòng nhập số từ 1 đến 8[/red]")
        except ValueError:
            console.print("[red]Vui lòng nhập số hợp lệ[/red]")
    
    example_map = {
        1: 'linear',
        2: 'multiple',
        3: 'polynomial',
        4: 'logistic',
        5: 'perceptron',
        6: 'neural',
        7: 'tf_neural',
        8: 'decision_tree'
    }
    return example_map[example_choice]

def run_example(example):
    """Chạy ví dụ được chọn"""
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
            "[bold cyan]Ví dụ 3: Hồi quy đa thức[/bold cyan]\n"
            "Dự đoán giá nhà với các mô hình đa thức khác nhau",
            border_style="cyan"
        ))
        polynomial_main()
    
    if example in ['logistic', 'all']:
        console.print(Panel(
            "[bold cyan]Ví dụ 4: Hồi quy logistic[/bold cyan]\n"
            "Phân loại học sinh đỗ/trượt dựa trên điểm thi",
            border_style="cyan"
        ))
        run_logistic_example()

    if example in ['perceptron', 'all']:
        console.print(Panel(
            "[bold cyan]Ví dụ 5: Perceptron[/bold cyan]\n"
            "Học hàm AND với perceptron",
            border_style="cyan"
        ))
        train_perceptron()
    
    if example in ['neural', 'all']:
        console.print(Panel(
            "[bold cyan]Ví dụ 6: Neural Network[/bold cyan]\n"
            "Học hàm XOR với neural network một lớp ẩn",
            border_style="cyan"
        ))
        run_neural_network()
    
    if example in ['tf_neural', 'all']:
        console.print(Panel(
            "[bold cyan]Ví dụ 7: TensorFlow Neural Network[/bold cyan]\n"
            "Học hàm AND với neural network sử dụng TensorFlow",
            border_style="cyan"
        ))
        run_tf_neural_network()

    if example in ['decision_tree', 'all']:
        console.print(Panel(
            "[bold cyan]Ví dụ 8: Decision Tree[/bold cyan]\n"
            "Phân loại hoa Iris sử dụng Decision Tree",
            border_style="cyan"
        ))
        from .examples.decision_tree.example import run_decision_tree_example
        run_decision_tree_example()

@click.command()
@click.option('--example', type=click.Choice([
    'linear', 'multiple', 'polynomial', 'logistic', 
    'perceptron', 'neural', 'tf_neural', 'decision_tree'
]), help='Chọn ví dụ để chạy')
def main(example):
    """Chạy các ví dụ về machine learning"""
    ensure_images_dir()
    print_examples_table()
    
    # Nếu không có lựa chọn từ command line, hỏi người dùng
    if example is None:
        example = get_user_choice()
    
    run_example(example)
    print_generated_images()

    if example == 'decision_tree':
        console.print("\n[bold green]Decision Tree Example[/bold green]")
        console.print("Phân loại hoa Iris sử dụng Decision Tree")
        console.print("Thuật toán xây dựng cây dựa trên:")
        console.print("1. Information Gain")
        console.print("2. Entropy")
        console.print("\nCông thức:")
        console.print("- Entropy: H(S) = -Σ p(x) * log2(p(x))")
        console.print("- Information Gain: IG(S,A) = H(S) - Σ |Sv|/|S| * H(Sv)")
        
        print_images(['decision_tree_boundary.png'])

def print_images(image_names):
    """In thông tin về các hình ảnh được chỉ định"""
    for img_name in image_names:
        img_path = os.path.join('images', img_name)
        if os.path.exists(img_path):
            size = os.path.getsize(img_path) / 1024  # Convert to KB
            console.print(f"📄 {img_name} ({size:.1f}KB)")
        else:
            console.print(f"❌ {img_name} (không tìm thấy)")

if __name__ == "__main__":
    main() 