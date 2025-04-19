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
from .examples.sklearn.linear_regression import main as run_sklearn_linear
from .examples.sklearn.multiple_linear_regression import main as run_sklearn_multiple

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
    },
    "Scikit-learn": {
        "sklearn_linear_regression.png": "Kết quả hồi quy tuyến tính với scikit-learn",
        "sklearn_multiple_linear_regression.png": "Feature importance của hồi quy nhiều biến với scikit-learn"
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
    """Print a table of available examples"""
    table = Table(title="Available Examples")
    table.add_column("Number", style="cyan")
    table.add_column("Example", style="green")
    table.add_column("Description", style="yellow")
    
    examples = [
        ("1", "Linear Regression", "Simple linear regression with one feature"),
        ("2", "Multiple Linear Regression", "Multiple linear regression with multiple features"),
        ("3", "Polynomial Regression", "Polynomial regression with regularization"),
        ("4", "Logistic Regression", "Binary classification with logistic regression"),
        ("5", "Perceptron", "Simple neural network for AND function"),
        ("6", "Neural Network", "Neural network with one hidden layer"),
        ("7", "TensorFlow Neural Network", "Neural network using TensorFlow"),
        ("8", "Decision Tree", "Decision tree for classification"),
        ("9", "Scikit-learn Linear Regression", "Linear regression using scikit-learn"),
        ("10", "Scikit-learn Multiple Linear Regression", "Multiple linear regression using scikit-learn")
    ]
    
    for number, example, description in examples:
        table.add_row(number, example, description)
    
    console.print(table)

def get_user_choice():
    """Get user's choice of example to run"""
    while True:
        try:
            choice = int(console.input("\n[bold cyan]Enter example number (1-10): [/bold cyan]"))
            if 1 <= choice <= 10:
                break
            console.print("[red]Please enter a number between 1 and 10[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")
    
    example_map = {
        1: 'linear',
        2: 'multiple',
        3: 'polynomial',
        4: 'logistic',
        5: 'perceptron',
        6: 'neural',
        7: 'tf_neural',
        8: 'decision_tree',
        9: 'sklearn',
        10: 'sklearn_multiple'
    }
    return example_map[choice]

def run_example(example):
    """Run the selected example"""
    if example == 'linear':
        run_linear_example()
    elif example == 'multiple':
        run_multiple_example()
    elif example == 'polynomial':
        run_polynomial_example()
    elif example == 'logistic':
        run_logistic_example()
    elif example == 'perceptron':
        run_perceptron_example()
    elif example == 'neural':
        run_neural_network_example()
    elif example == 'tf_neural':
        run_tf_neural_network_example()
    elif example == 'decision_tree':
        run_decision_tree_example()
    elif example == 'sklearn':
        run_sklearn_linear()
    elif example == 'sklearn_multiple':
        run_sklearn_multiple()

@click.command()
@click.option('--example', type=click.Choice(['linear', 'multiple', 'polynomial', 'logistic', 'perceptron', 'neural', 'tf_neural', 'decision_tree', 'sklearn', 'sklearn_multiple']), help='Example to run')
def main(example):
    """Chạy các ví dụ về machine learning"""
    ensure_images_dir()
    print_examples_table()
    
    # Nếu không có lựa chọn từ command line, hỏi người dùng
    if example is None:
        example = get_user_choice()
    
    run_example(example)
    print_generated_images()

if __name__ == "__main__":
    main() 