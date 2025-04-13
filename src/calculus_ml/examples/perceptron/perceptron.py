"""
Triển khai perceptron đơn giản sử dụng hàm kích hoạt sigmoid.
Perceptron là đơn vị cơ bản của mạng neural, có thể học các hàm logic đơn giản.
"""

import numpy as np

class Perceptron:
    """
    Lớp Perceptron triển khai một neuron đơn giản với hàm kích hoạt sigmoid.
    
    Attributes:
        weights (numpy.ndarray): Vector trọng số
        bias (float): Độ chệch
        lr (float): Tốc độ học
        history (dict): Lịch sử training, bao gồm loss
    """
    
    def __init__(self, input_dim, lr=0.01):
        """
        Khởi tạo perceptron với số chiều đầu vào và tốc độ học.
        
        Args:
            input_dim (int): Số chiều của vector đầu vào
            lr (float): Tốc độ học, mặc định là 0.01
        """
        # Khởi tạo trọng số ngẫu nhiên với phân phối chuẩn
        self.weights = np.random.randn(input_dim)
        # Khởi tạo độ chệch bằng 0
        self.bias = 0
        # Lưu tốc độ học
        self.lr = lr
        # Khởi tạo lịch sử training
        self.history = {'loss': []}
    
    def sigmoid(self, z):
        """
        Hàm kích hoạt sigmoid.
        
        Args:
            z (float): Giá trị đầu vào
            
        Returns:
            float: Giá trị đầu ra của hàm sigmoid
        """
        return 1 / (1 + np.exp(-z))
    
    def forward(self, x):
        """
        Thực hiện lan truyền tiến (forward propagation).
        
        Args:
            x (numpy.ndarray): Vector đầu vào
            
        Returns:
            float: Giá trị dự đoán
        """
        # Tính tổng có trọng số
        z = np.dot(x, self.weights) + self.bias
        # Áp dụng hàm sigmoid
        return self.sigmoid(z)
    
    def backward(self, x, y, y_hat):
        """
        Thực hiện lan truyền ngược (backward propagation) và cập nhật tham số.
        
        Args:
            x (numpy.ndarray): Vector đầu vào
            y (float): Giá trị thực tế
            y_hat (float): Giá trị dự đoán
        """
        # Tính đạo hàm của loss function theo các tham số
        error = y_hat - y
        # Cập nhật trọng số
        self.weights -= self.lr * error * x
        # Cập nhật độ chệch
        self.bias -= self.lr * error
    
    def compute_loss(self, X, y):
        """
        Tính loss function cho toàn bộ tập dữ liệu.
        
        Args:
            X (numpy.ndarray): Ma trận đầu vào
            y (numpy.ndarray): Vector nhãn
            
        Returns:
            float: Giá trị loss trung bình
        """
        total_loss = 0
        for i in range(len(X)):
            y_hat = self.forward(X[i])
            # Binary cross-entropy loss
            loss = -(y[i] * np.log(y_hat) + (1 - y[i]) * np.log(1 - y_hat))
            total_loss += loss
        return total_loss / len(X)
    
    def train(self, X, y, epochs=1000):
        """
        Huấn luyện perceptron trên tập dữ liệu.
        
        Args:
            X (numpy.ndarray): Ma trận đầu vào
            y (numpy.ndarray): Vector nhãn
            epochs (int): Số lần lặp huấn luyện
        """
        for epoch in range(epochs):
            for i in range(len(X)):
                # Lấy mẫu dữ liệu
                x_i = X[i]
                y_i = y[i]
                
                # Lan truyền tiến
                y_hat = self.forward(x_i)
                
                # Lan truyền ngược và cập nhật
                self.backward(x_i, y_i, y_hat)
            
            # Tính loss sau mỗi epoch
            loss = self.compute_loss(X, y)
            
            # Lưu vào lịch sử
            self.history['loss'].append(loss)
            
            # In thông tin mỗi 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")
    
    def predict(self, x):
        """
        Dự đoán đầu ra cho một mẫu dữ liệu.
        
        Args:
            x (numpy.ndarray): Vector đầu vào
            
        Returns:
            float: Xác suất dự đoán
        """
        return self.forward(x)