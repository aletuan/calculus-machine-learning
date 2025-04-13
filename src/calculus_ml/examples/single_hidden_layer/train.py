"""
Huấn luyện mạng neural với dữ liệu XOR.
"""

import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from .model_numpy import NeuralNetwork
from .activations import sigmoid, sigmoid_derivative

console = Console()

def main():
    # Dữ liệu XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Khởi tạo mạng neural
    input_size = 2
    hidden_size = 4
    output_size = 1
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    
    # Hiển thị quá trình training
    console.print("\n[bold yellow]Quá trình training:[/bold yellow]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Training...", total=10000)
        
        def training_callback(epoch, loss):
            progress.update(task, advance=1)
            if epoch % 1000 == 0 or epoch == 9999:
                console.print(f"Epoch {epoch:4d}: Loss = {loss:.4f}")
        
        # Huấn luyện mạng neural
        nn.train(X, y, epochs=10000, callback=training_callback)

    # Vẽ đồ thị quá trình training
    plt.figure(figsize=(8, 5))
    plt.plot(nn.history['loss'])
    plt.title('Loss qua các epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Lưu đồ thị
    plt.tight_layout()
    plt.savefig('images/neural_network_training_history.png')
    plt.close()

    # Tạo nội dung cho panel kết quả
    results = [
        "[bold green]Kết quả huấn luyện:[/bold green]",
        "Input | Target | Prediction",
        "-" * 35
    ]
    
    # Kiểm tra mô hình trên tất cả các đầu vào có thể
    predictions = nn.predict(X)
    for i in range(len(X)):
        x_str = f"({X[i][0]}, {X[i][1]})"
        y_str = f"{y[i][0]}"
        pred_str = f"{predictions[i][0]:.4f}"
        results.append(f"{x_str:8} | {y_str:6} | {pred_str}")
    
    # Hiển thị kết quả trong panel
    console.print(Panel(
        "\n".join(results),
        title="Neural Network Results",
        border_style="cyan"
    ))
    
    console.print("[green]✓[/green] Neural network visualization saved")

if __name__ == "__main__":
    main()