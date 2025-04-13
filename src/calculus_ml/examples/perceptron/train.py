"""
Huấn luyện perceptron học hàm AND.
Ví dụ này minh họa cách một mạng neural đơn giản có thể học các hàm logic cơ bản.
"""

import matplotlib.pyplot as plt
from .perceptron import Perceptron
from .and_dataset import load_and_data

def train_perceptron():
    """Huấn luyện perceptron học hàm AND"""
    # Tải dữ liệu hàm AND
    X, y = load_and_data()

    # Khởi tạo và huấn luyện perceptron
    model = Perceptron(input_dim=2, lr=0.1)  # 2 đặc trưng đầu vào, learning rate 0.1
    model.train(X, y, epochs=1000)           # Huấn luyện trong 1000 epochs

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

    # Kiểm tra mô hình trên tất cả các đầu vào có thể
    print("\nKiểm tra perceptron đã được huấn luyện:")
    print("Đầu vào (x1, x2) | Dự đoán | Nhãn")
    print("-" * 35)
    for x_i, y_i in zip(X, y):
        y_hat = model.predict(x_i)
        print(f"({x_i[0]}, {x_i[1]})        | {y_hat:.4f}    | {y_i}")

if __name__ == "__main__":
    train_perceptron()