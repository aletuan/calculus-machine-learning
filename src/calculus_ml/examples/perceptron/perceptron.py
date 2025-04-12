import numpy as np

class Perceptron:
    """Perceptron model for binary classification.
    
    This implementation uses sigmoid activation function and gradient descent
    for training. It can be used to learn simple logical functions like AND, OR.
    
    Attributes:
        w (ndarray): Weight vector of shape (input_dim,)
        b (float): Bias term
        lr (float): Learning rate for gradient descent
    """
    
    def __init__(self, input_dim, lr=0.1):
        """Initialize the perceptron.
        
        Args:
            input_dim (int): Number of input features
            lr (float): Learning rate for gradient descent (default: 0.1)
        """
        self.w = np.zeros(input_dim)  # Initialize weights to zeros
        self.b = 0.0                  # Initialize bias to zero
        self.lr = lr                  # Learning rate

    def sigmoid(self, z):
        """Sigmoid activation function.
        
        Args:
            z (float): Linear combination of inputs
            
        Returns:
            float: Probability between 0 and 1
        """
        return 1 / (1 + np.exp(-z))

    def predict(self, x):
        """Make a prediction for input x.
        
        Args:
            x (ndarray): Input vector of shape (input_dim,)
            
        Returns:
            float: Predicted probability between 0 and 1
        """
        z = np.dot(self.w, x) + self.b  # Linear combination
        return self.sigmoid(z)           # Apply sigmoid activation

    def train(self, X, y, epochs=100):
        """Train the perceptron using gradient descent.
        
        Args:
            X (ndarray): Training data of shape (n_samples, input_dim)
            y (ndarray): Target values of shape (n_samples,)
            epochs (int): Number of training epochs (default: 100)
        """
        for epoch in range(epochs):
            for i in range(len(X)):
                x_i = X[i]              # Get current sample
                y_i = y[i]              # Get current label
                y_hat = self.predict(x_i)  # Make prediction
                error = y_hat - y_i      # Compute error
                
                # Update weights and bias using gradient descent
                self.w -= self.lr * error * x_i  # Update weights
                self.b -= self.lr * error        # Update bias