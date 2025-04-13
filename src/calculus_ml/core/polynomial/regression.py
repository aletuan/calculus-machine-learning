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
        self.feature_means = None
        self.feature_stds = None
        
    def _generate_polynomial_features(self, X):
        """Generate polynomial features up to the specified degree."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape
        
        # Standardize X first
        if self.feature_means is None:  # Training phase
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0) + 1e-8
        
        X_scaled = (X - self.feature_means) / self.feature_stds
        X_poly = np.ones((n_samples, 1))
        
        for d in range(1, self.degree + 1):
            X_poly = np.column_stack((X_poly, X_scaled ** d))
            
        return X_poly[:, 1:]  # Remove the constant term
        
    def predict(self, X):
        """Make predictions using the trained model."""
        X_poly = self._generate_polynomial_features(X)
        return np.clip(np.dot(X_poly, self.weights) + self.bias, 0, None)  # Ensure non-negative predictions
    
    def compute_cost(self, X, y):
        """Compute cost with regularization."""
        m = len(y)
        y_pred = self.predict(X)
        
        # Compute MSE in a numerically stable way
        squared_errors = np.clip((y_pred - y)**2, 0, 1e10)
        mse = np.mean(squared_errors)
        
        # Compute regularization term with scaling
        reg_term = self.lambda_reg * np.mean(self.weights**2) / (2 * m)
        
        return mse/2 + reg_term
    
    def compute_gradient(self, X, y):
        """Compute gradients with regularization."""
        m = len(y)
        X_poly = self._generate_polynomial_features(X)
        y_pred = self.predict(X)
        
        # Compute gradients with numerical stability
        errors = y_pred - y
        dw = np.mean(X_poly * errors[:, np.newaxis], axis=0)
        dw += (self.lambda_reg * self.weights) / m
        
        db = np.mean(errors)
        
        # Clip gradients to prevent explosion
        dw = np.clip(dw, -1e10, 1e10)
        db = np.clip(db, -1e10, 1e10)
        
        return dw, db

    def fit(self, X, y):
        """Fit the polynomial regression model."""
        # Generate polynomial features
        X_poly = self._generate_polynomial_features(X)
        
        # Initialize parameters with small random values
        self.weights = np.random.randn(X_poly.shape[1]) * 0.01
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        m = len(y)
        for i in range(self.num_iterations):
            # Forward pass
            y_pred = self.predict(X)
            
            # Compute gradients
            dw, db = self.compute_gradient(X, y)
            
            # Update parameters with gradient clipping
            self.weights -= np.clip(self.learning_rate * dw, -1.0, 1.0)
            self.bias -= np.clip(self.learning_rate * db, -1.0, 1.0)
            
            # Compute cost
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            
            # Early stopping if cost is very small or not changing
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