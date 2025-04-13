"""
Triển khai mạng neural với một lớp ẩn sử dụng NumPy.
Mạng này có thể học các hàm logic phức tạp hơn như XOR.
"""

import numpy as np
from .activations import sigmoid, sigmoid_derivative

class NeuralNetwork:
    """
    Mạng neural với một lớp ẩn.
    
    Attributes:
        input_size (int): Số neuron đầu vào
        hidden_size (int): Số neuron lớp ẩn
        output_size (int): Số neuron đầu ra
        W1 (numpy.ndarray): Ma trận trọng số lớp ẩn
        b1 (numpy.ndarray): Vector độ chệch lớp ẩn
        W2 (numpy.ndarray): Ma trận trọng số lớp đầu ra
        b2 (numpy.ndarray): Vector độ chệch lớp đầu ra
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Khởi tạo mạng neural với các tham số.
        
        Args:
            input_size (int): Số neuron đầu vào
            hidden_size (int): Số neuron lớp ẩn
            output_size (int): Số neuron đầu ra
        """
        # Khởi tạo kích thước các lớp
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Khởi tạo trọng số ngẫu nhiên với phân phối chuẩn
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
    
    def forward(self, X):
        """
        Thực hiện lan truyền tiến.
        
        Args:
            X (numpy.ndarray): Ma trận đầu vào
            
        Returns:
            tuple: (Z1, A1, Z2, A2) - Các giá trị trung gian và đầu ra
        """
        # Lan truyền từ lớp đầu vào đến lớp ẩn
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = sigmoid(Z1)
        
        # Lan truyền từ lớp ẩn đến lớp đầu ra
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = sigmoid(Z2)
        
        return Z1, A1, Z2, A2
    
    def backward(self, X, y, Z1, A1, Z2, A2):
        """
        Thực hiện lan truyền ngược và cập nhật tham số.
        
        Args:
            X (numpy.ndarray): Ma trận đầu vào
            y (numpy.ndarray): Vector nhãn
            Z1 (numpy.ndarray): Giá trị trước khi kích hoạt lớp ẩn
            A1 (numpy.ndarray): Giá trị sau khi kích hoạt lớp ẩn
            Z2 (numpy.ndarray): Giá trị trước khi kích hoạt lớp đầu ra
            A2 (numpy.ndarray): Giá trị sau khi kích hoạt lớp đầu ra
        """
        # Tính đạo hàm của loss function
        dZ2 = A2 - y
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_derivative(Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # Cập nhật tham số
        learning_rate = 0.1
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def train(self, X, y, epochs=10000):
        """
        Huấn luyện mạng neural.
        
        Args:
            X (numpy.ndarray): Ma trận đầu vào
            y (numpy.ndarray): Vector nhãn
            epochs (int): Số lần lặp huấn luyện
        """
        for epoch in range(epochs):
            # Lan truyền tiến
            Z1, A1, Z2, A2 = self.forward(X)
            
            # Lan truyền ngược và cập nhật
            self.backward(X, y, Z1, A1, Z2, A2)
            
            # In loss mỗi 1000 epochs
            if epoch % 1000 == 0:
                loss = np.mean(np.square(A2 - y))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """
        Dự đoán đầu ra cho một mẫu dữ liệu.
        
        Args:
            X (numpy.ndarray): Ma trận đầu vào
            
        Returns:
            numpy.ndarray: Giá trị dự đoán
        """
        _, _, _, A2 = self.forward(X)
        return A2