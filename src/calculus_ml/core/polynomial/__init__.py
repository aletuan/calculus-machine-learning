"""
Polynomial Regression module with regularization support.
"""

import numpy as np
from ..base.model import RegressionModel

class PolynomialRegression(RegressionModel):
    """Polynomial Regression with regularization.
    
    Parameters
    ----------
    degree : int
        The degree of the polynomial features
    lambda_reg : float
        Regularization strength (default: 0.0)
    learning_rate : float
        Learning rate for gradient descent (default: 0.01)
    num_iterations : int
        Maximum number of iterations for gradient descent (default: 1000)
    """
    
    def __init__(self, degree=2, lambda_reg=0.0, learning_rate=0.01, num_iterations=1000):
        super().__init__(learning_rate, num_iterations)
        self.degree = degree
        self.lambda_reg = lambda_reg
        
    def _generate_polynomial_features(self, X):
        """Generate polynomial features up to the specified degree."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape
        X_poly = np.ones((n_samples, 1))
        
        # Scale X to prevent overflow
        X_scaled = X / np.max(np.abs(X))
        
        for d in range(1, self.degree + 1):
            X_poly = np.column_stack((X_poly, X_scaled ** d))
            
        return X_poly[:, 1:]  # Remove the constant term
        
    def predict(self, X):
        """Make predictions using the trained model."""
        X_poly = self._generate_polynomial_features(X)
        return np.dot(X_poly, self.weights) + self.bias
    
    def compute_cost(self, X, y):
        """Compute cost with regularization."""
        m = len(y)
        y_pred = self.predict(X)
        
        # Compute MSE in a numerically stable way
        squared_errors = np.clip((y_pred - y)**2, 0, 1e10)  # Prevent overflow
        mse = np.mean(squared_errors)
        
        # Compute regularization term
        reg_term = self.lambda_reg * np.mean(np.clip(self.weights**2, 0, 1e10))
        
        return mse/2 + reg_term/2
    
    def compute_gradient(self, X, y):
        """Compute gradients with regularization."""
        m = len(y)
        X_poly = self._generate_polynomial_features(X)
        y_pred = self.predict(X)
        
        # Clip predictions to prevent overflow
        errors = np.clip(y_pred - y, -1e10, 1e10)
        
        # Compute gradients with numerical stability
        dw = np.mean(X_poly * errors[:, np.newaxis], axis=0)
        dw += self.lambda_reg * np.clip(self.weights, -1e10, 1e10) / m
        
        db = np.mean(errors)
        
        return dw, db

    def fit(self, X, y):
        """Fit the polynomial regression model."""
        # Generate polynomial features
        X_poly = self._generate_polynomial_features(X)
        
        # Initialize parameters
        self.weights = np.zeros(X_poly.shape[1])
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        m = len(y)
        for i in range(self.num_iterations):
            # Forward pass
            y_pred = self.predict(X)
            
            # Compute gradients with regularization
            dw, db = self.compute_gradient(X, y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute cost with regularization
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            
            # Early stopping if cost change is very small
            if i > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < 1e-8:
                break
                
    def _generate_polynomial_features(self, X):
        """Generate polynomial features up to the specified degree."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape
        X_poly = np.ones((n_samples, 1))
        
        for d in range(1, self.degree + 1):
            X_poly = np.column_stack((X_poly, X ** d))
            
        return X_poly[:, 1:]  # Remove the constant term
        
    def predict(self, X):
        """Make predictions using the trained model."""
        X_poly = self._generate_polynomial_features(X)
        return np.dot(X_poly, self.weights) + self.bias 