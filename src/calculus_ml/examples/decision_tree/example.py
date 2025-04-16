import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from ...core.decision_tree.tree import DecisionTree

console = Console()

def generate_data():
    """
    Tạo dữ liệu mẫu cho bài toán phân loại
    Dataset: Phân loại hoa iris (đơn giản hóa thành 2 class)
    """
    # Load dữ liệu iris và chỉ lấy 2 class đầu tiên
    iris = load_iris()
    X = iris.data[:100, [0, 1]]  # Chỉ lấy 2 features đầu tiên
    y = iris.target[:100]        # Chỉ lấy 2 classes đầu tiên
    
    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def plot_decision_boundary(X, y, model):
    """
    Vẽ decision boundary của model
    """
    h = 0.02  # Kích thước grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Dự đoán cho toàn bộ grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Vẽ contour
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.colorbar()

def run_decision_tree_example():
    """
    Ví dụ về Decision Tree với dataset Iris
    """
    console.print("\n[bold cyan]Decision Tree Example[/bold cyan]", justify="center")
    
    # Hiển thị công thức
    console.print(Panel(
        "[bold green]Công thức tính toán:[/bold green]\n"
        "1. Entropy:\n"
        "   H(S) = -Σ p(x) * log2(p(x))\n"
        "2. Information Gain:\n"
        "   IG(S,A) = H(S) - Σ |Sv|/|S| * H(Sv)\n"
        "3. Điều kiện dừng:\n"
        "   - Đạt độ sâu tối đa (max_depth)\n"
        "   - Số mẫu nhỏ hơn ngưỡng (min_samples_split)",
        title="Decision Tree Formulas",
        border_style="cyan"
    ))
    
    # Tạo dữ liệu
    X_train, X_test, y_train, y_test = generate_data()
    
    # Hiển thị thông tin dữ liệu
    console.print(Panel(
        f"[bold green]Dữ liệu mẫu:[/bold green]\n"
        f"- Dataset: Iris (2 class đầu tiên)\n"
        f"- Features: Sepal length, Sepal width\n"
        f"- Labels: 0 (setosa), 1 (versicolor)\n"
        f"- Số mẫu huấn luyện: {X_train.shape[0]}\n"
        f"- Số mẫu kiểm tra: {X_test.shape[0]}\n"
        f"- Số features: {X_train.shape[1]}",
        title="Training Data",
        border_style="cyan"
    ))
    
    # Khởi tạo và huấn luyện mô hình
    console.print("\n[bold yellow]Quá trình training:[/bold yellow]")
    tree = DecisionTree(max_depth=3, min_samples_split=2)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Đang xây dựng cây quyết định...", total=100)
        tree.fit(X_train, y_train)
        progress.update(task, completed=100)
    
    # Dự đoán và đánh giá
    y_pred = tree.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    # Hiển thị kết quả
    console.print(Panel(
        f"[bold green]Kết quả tối ưu:[/bold green]\n"
        f"1. Hiệu năng mô hình:\n"
        f"   - Độ chính xác trên tập test: {accuracy:.2f}\n"
        f"2. Cấu trúc cây:\n"
        f"   - Độ sâu tối đa: {tree.max_depth}\n"
        f"   - Số mẫu tối thiểu để split: {tree.min_samples_split}",
        title="Model Results",
        border_style="cyan"
    ))
    
    # Trực quan hóa
    console.print("\n[bold yellow]Trực quan hóa kết quả:[/bold yellow]")
    plt.figure(figsize=(10, 5))
    
    # Vẽ dữ liệu huấn luyện và decision boundary
    plt.subplot(121)
    plot_decision_boundary(X_train, y_train, tree)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
    plt.title('Decision Boundary trên tập train')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    
    # Vẽ dữ liệu test và dự đoán
    plt.subplot(122)
    plot_decision_boundary(X_test, y_test, tree)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')
    plt.title('Dự đoán trên tập test')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    
    plt.tight_layout()
    plt.savefig('images/decision_tree_boundary.png')
    plt.close()
    
    console.print("[green]✓[/green] Decision tree visualization saved as 'images/decision_tree_boundary.png'")

if __name__ == "__main__":
    run_decision_tree_example() 