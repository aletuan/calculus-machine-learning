import numpy as np

class StandardScaler:
    """Chuẩn hóa dữ liệu bằng phương pháp Standard Scaling"""
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X):
        """Tính mean và std từ dữ liệu training"""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Tránh chia cho 0
        self.std[self.std == 0] = 1
        return self
    
    def transform(self, X):
        """Chuẩn hóa dữ liệu"""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler chưa được fit")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        """Fit và transform dữ liệu"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """Chuyển dữ liệu đã chuẩn hóa về dạng gốc"""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler chưa được fit")
        return X_scaled * self.std + self.mean 