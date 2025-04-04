import numpy as np
from ..base.model import RegressionModel

class LinearRegression(RegressionModel):
    """Linear Regression model"""
    
    def predict(self, X):
        """Dự đoán giá trị y = w*X + b"""
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return np.dot(X, self.weights) + self.bias
    
    def compute_cost(self, X, y):
        """Tính MSE cost function"""
        m = len(y)
        predictions = self.predict(X)
        return (1/(2*m)) * np.sum((predictions - y)**2)
    
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