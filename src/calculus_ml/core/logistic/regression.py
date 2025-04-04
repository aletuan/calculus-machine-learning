import numpy as np
from ..base.model import RegressionModel

class LogisticRegression(RegressionModel):
    """Logistic Regression model"""
    
    def sigmoid(self, z):
        """Hàm sigmoid"""
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X):
        """Dự đoán xác suất P(y=1)"""
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def compute_cost(self, X, y):
        """Binary cross-entropy loss"""
        m = len(y)
        predictions = self.predict(X)
        epsilon = 1e-15  # Tránh log(0)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -(1/m) * np.sum(y*np.log(predictions) + (1-y)*np.log(1-predictions))
    
    def compute_gradient(self, X, y):
        """Tính gradient của cost function"""
        m = len(y)
        predictions = self.predict(X)
        error = predictions - y
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        dj_dw = (1/m) * np.dot(X.T, error)
        dj_db = (1/m) * np.sum(error)
        
        return dj_dw, dj_db 