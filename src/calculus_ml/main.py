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
from .examples.perceptron.train import train_perceptron
from .examples.single_hidden_layer.train import main as run_neural_network

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
        "perceptron_decision_boundary.png": "Decision boundary của perceptron",
        "perceptron_training_history.png": "Lịch sử training của perceptron"
    },
    "Neural Network": {
        "neural_network_decision_boundary.png": "Decision boundary của neural network",
        "neural_network_training_history.png": "Lịch sử training của neural network"
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
@click.option('--example', type=click.Choice(['linear', 'multiple', 'polynomial', 'logistic', 'perceptron', 'neural', 'all']), 
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

    if example in ['perceptron', 'all']:
        console.print(Panel(
            "[bold cyan]Ví dụ 5: Perceptron[/bold cyan]\n"
            "Huấn luyện perceptron học hàm AND",
            border_style="cyan"
        ))
        
        console.print(Panel(
            "[bold yellow]Công thức Perceptron[/bold yellow]\n"
            "1. Hàm kích hoạt (Activation function):\n"
            "   f(z) = 1 / (1 + e^(-z))  (Sigmoid)\n\n"
            "2. Hàm dự đoán (Prediction):\n"
            "   y_hat = f(w₁x₁ + w₂x₂ + b)\n\n"
            "3. Loss function:\n"
            "   L(y, y_hat) = -(y * log(y_hat) + (1-y) * log(1-y_hat))\n\n"
            "4. Gradient descent:\n"
            "   w₁ = w₁ - α * (y_hat - y) * x₁\n"
            "   w₂ = w₂ - α * (y_hat - y) * x₂\n"
            "   b = b - α * (y_hat - y)\n\n"
            "Trong đó:\n"
            "- w₁, w₂: trọng số (weights)\n"
            "- b: độ chệch (bias)\n"
            "- α: learning rate\n"
            "- x₁, x₂: đầu vào (input features)\n"
            "- y: nhãn thực tế (true label)\n"
            "- y_hat: dự đoán (prediction)",
            border_style="yellow"
        ))
        
        train_perceptron()

    if example in ['neural', 'all']:
        console.print(Panel(
            "[bold cyan]Ví dụ 6: Neural Network[/bold cyan]\n"
            "Huấn luyện mạng neural với một lớp ẩn để giải quyết bài toán XOR",
            border_style="cyan"
        ))
        
        console.print(Panel(
            "[bold yellow]Công thức Neural Network[/bold yellow]\n"
            "1. Hàm kích hoạt (Activation function):\n"
            "   f(z) = 1 / (1 + e^(-z))  (Sigmoid)\n\n"
            "2. Lan truyền tiến (Forward propagation):\n"
            "   Z1 = X·W1 + b1\n"
            "   A1 = f(Z1)\n"
            "   Z2 = A1·W2 + b2\n"
            "   A2 = f(Z2)\n\n"
            "3. Lan truyền ngược (Backward propagation):\n"
            "   dZ2 = A2 - Y\n"
            "   dW2 = A1ᵀ·dZ2\n"
            "   db2 = sum(dZ2)\n"
            "   dZ1 = dZ2·W2ᵀ * f'(Z1)\n"
            "   dW1 = Xᵀ·dZ1\n"
            "   db1 = sum(dZ1)\n\n"
            "4. Cập nhật tham số:\n"
            "   W2 = W2 - α·dW2\n"
            "   b2 = b2 - α·db2\n"
            "   W1 = W1 - α·dW1\n"
            "   b1 = b1 - α·db1\n\n"
            "Trong đó:\n"
            "- W1, W2: ma trận trọng số\n"
            "- b1, b2: vector độ chệch\n"
            "- α: learning rate\n"
            "- X: ma trận đầu vào\n"
            "- Y: vector nhãn thực tế\n"
            "- A2: dự đoán",
            border_style="yellow"
        ))
        
        run_neural_network()

    # In thông tin về các hình ảnh đã tạo
    print_generated_images()

    # Print image information
    print("\n\n╭────────────────────────────────────────── Thông tin hình ảnh ───────────────────────────────────────────╮")
    print("│ 📊 Hình ảnh đã tạo                                                                                      │")
    print("│ ├── 📁 Linear Regression                                                                                │")
    print("│ │   ├── 📄 linear_regression_fit.png (46.5KB) - Đường hồi quy tuyến tính và dữ liệu                     │")
    print("│ │   ├── 📄 linear_cost_history.png (26.2KB) - Lịch sử cost function của linear regression               │")
    print("│ │   ├── 📄 multiple_regression_fit.png (142.0KB) - Mặt phẳng hồi quy nhiều biến và dữ liệu              │")
    print("│ │   └── 📄 multiple_cost_history.png (26.7KB) - Lịch sử cost function của hồi quy nhiều biến            │")
    print("│ ├── 📁 Polynomial Regression                                                                            │")
    print("│ │   ├── 📄 house_price_data.png (size) - Dữ liệu giá nhà                                               │")
    print("│ │   ├── 📄 polynomial_regression_fit.png (160.1KB) - So sánh các mô hình polynomial khác bậc            │")
    print("│ │   └── 📄 regularization_effect.png (224.0KB) - Ảnh hưởng của regularization                           │")
    print("│ ├── 📁 Logistic Regression                                                                              │")
    print("│ │   ├── 📄 logistic_decision_boundary.png (37.4KB) - Decision boundary của logistic regression          │")
    print("│ │   └── 📄 logistic_cost_history.png (22.5KB) - Lịch sử cost function của logistic regression           │")
    print("│ ├── 📁 Perceptron                                                                                       │")
    print("│ │   ├── 📄 perceptron_decision_boundary.png - Decision boundary của perceptron                          │")
    print("│ │   └── 📄 perceptron_training_history.png - Lịch sử training của perceptron                            │")
    print("│ └── 📁 Neural Network                                                                                    │")
    print("│     ├── 📄 neural_network_decision_boundary.png - Decision boundary của neural network                    │")
    print("│     └── 📄 neural_network_training_history.png - Lịch sử training của neural network                    │")
    print("╰─────────────────────────────────────────────────────────────────────────────────────────────────────────╯")

if __name__ == "__main__":
    main() 