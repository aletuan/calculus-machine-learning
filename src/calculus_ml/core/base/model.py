from abc import ABC, abstractmethod
import numpy as np

class RegressionModel(ABC):
    """Abstract base class cho các mô hình regression"""
    
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.param_history = []
    
    @abstractmethod
    def predict(self, X):
        """Dự đoán giá trị y"""
        pass
    
    @abstractmethod
    def compute_cost(self, X, y):
        """Tính cost function"""
        pass
    
    @abstractmethod
    def compute_gradient(self, X, y):
        """Tính gradient của cost function"""
        pass
    
    def fit(self, X, y):
        """Thực hiện gradient descent để tối ưu tham số"""
        # Khởi tạo tham số
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.num_iterations):
            # Tính gradient
            dj_dw, dj_db = self.compute_gradient(X, y)
            
            # Cập nhật tham số
            self.weights = self.weights - self.learning_rate * dj_dw
            self.bias = self.bias - self.learning_rate * dj_db
            
            # Lưu lịch sử
            if i < 100000:
                self.cost_history.append(self.compute_cost(X, y))
                self.param_history.append([self.weights.copy(), self.bias])
        
        return {
            'cost_history': self.cost_history,
            'param_history': self.param_history
        } 