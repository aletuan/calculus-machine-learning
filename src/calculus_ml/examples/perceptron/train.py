"""
Huấn luyện perceptron học hàm AND.
Ví dụ này minh họa cách một mạng neural đơn giản có thể học các hàm logic cơ bản.
"""

import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from .perceptron import Perceptron
from .and_dataset import load_and_data

console = Console()

def train_perceptron():
    """Huấn luyện perceptron học hàm AND"""
    # Tải dữ liệu hàm AND
    X, y = load_and_data()

    # Khởi tạo perceptron
    model = Perceptron(input_dim=2, lr=0.1)  # 2 đặc trưng đầu vào, learning rate 0.1
    
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
        
        def training_callback(epoch, loss):
            progress.update(task, advance=1)
            if epoch % 100 == 0 or epoch == 999:
                console.print(f"Epoch {epoch:4d}: Loss = {loss:.4f}")
        
        # Huấn luyện perceptron
        model.train(X, y, epochs=1000, callback=training_callback)

    # Vẽ đồ thị quá trình training
    plt.figure(figsize=(8, 5))
    plt.plot(model.history['loss'])
    plt.title('Loss qua các epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Lưu đồ thị
    plt.tight_layout()
    plt.savefig('images/perceptron_training_history.png')
    plt.close()

    # Tạo nội dung cho panel kết quả
    results = [
        "[bold green]Kết quả huấn luyện:[/bold green]",
        "Đầu vào (x1, x2) | Dự đoán | Nhãn",
        "-" * 35
    ]
    
    # Kiểm tra mô hình trên tất cả các đầu vào có thể
    for x_i, y_i in zip(X, y):
        y_hat = model.predict(x_i)
        results.append(f"({x_i[0]}, {x_i[1]})        | {y_hat:.4f}    | {y_i}")
    
    # Hiển thị kết quả trong panel
    console.print(Panel(
        "\n".join(results),
        title="Perceptron Results",
        border_style="cyan"
    ))
    
    console.print("[green]✓[/green] Perceptron visualization saved")

if __name__ == "__main__":
    train_perceptron()