"""
Example of Polynomial Regression with Regularization.
"""

import numpy as np
from ..core.polynomial import PolynomialRegression
from ..visualization.polynomial import (
    plot_polynomial_comparison,
    plot_regularization_effect
)
import matplotlib.pyplot as plt

def generate_nonlinear_data(n_samples=100, noise=0.3):
    """Generate synthetic nonlinear data."""
    np.random.seed(42)
    # Generate positive X values between 1 and 5 (1000 sqft)
    X = np.random.uniform(1.0, 5.0, n_samples)
    # Generate y with stronger polynomial relationship
    y = 50 + 100*X + 20*X**2 + 5*X**3
    y += noise * np.random.randn(n_samples) * 100  # Scale noise appropriately
    # Ensure non-negative house prices
    y = np.clip(y, 0, None)
    return X, y

def run_polynomial_comparison():
    """Run and visualize polynomial regression with different degrees."""
    # Generate data
    X, y = generate_nonlinear_data()
    
    # Train models with different degrees
    degrees = [1, 2, 4]
    models = []
    
    for degree in degrees:
        model = PolynomialRegression(degree=degree, lambda_reg=0.0,
                                   learning_rate=0.01, num_iterations=1000)
        model.fit(X, y)
        models.append(model)
    
    # Plot comparison
    plot_polynomial_comparison(
        X, y, models, degrees,
        save_path='images/polynomial_regression_fit.png'
    )
    
    return models[2]  # Return degree 4 model for regularization example

def run_regularization_example(base_model):
    """Run and visualize the effect of regularization."""
    # Generate data
    X, y = generate_nonlinear_data(noise=0.5)  # More noise
    
    # Train models with different lambda values - using smaller values
    lambdas = [0.0, 0.01, 1.0]  # Changed from [0.0, 0.1, 10.0]
    models = []
    
    # Sort X for consistent testing
    X_test = np.linspace(1.0, 5.0, 10)
    print("\nTesting predictions for different lambda values:")
    print("X (diện tích) | λ=0.0 | λ=0.01 | λ=1.0")
    print("-" * 45)
    
    # Train models and store predictions
    predictions = []
    for lambda_reg in lambdas:
        model = PolynomialRegression(degree=4, lambda_reg=lambda_reg,
                                   learning_rate=0.01, num_iterations=1000)
        model.fit(X, y)
        models.append(model)
        predictions.append(model.predict(X_test))
    
    # Print predictions for inspection
    for i, x in enumerate(X_test):
        print(f"{x:12.1f} | {predictions[0][i]:6.0f} | {predictions[1][i]:7.0f} | {predictions[2][i]:6.0f}")
    
    # Plot regularization effect
    plot_regularization_effect(
        X, y, models, lambdas,
        save_path='images/regularization_effect.png'
    )

def plot_house_price_data(X, y, save_path='images/house_price_data.png'):
    """Plot house price data.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, 1)
        House sizes in 1000 sqft
    y : ndarray of shape (n_samples,)
        House prices in 1000$
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.5, label='House Data')
    plt.xlabel('Diện tích (1000 sqft)')
    plt.ylabel('Giá nhà (1000$)')
    plt.title('Dữ liệu giá nhà')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run all polynomial regression examples."""
    print("Running polynomial regression examples...")
    
    # Generate and plot house price data
    X, y = generate_nonlinear_data()
    plot_house_price_data(X, y)
    print("Generated house_price_data.png")
    
    # Run polynomial comparison
    base_model = run_polynomial_comparison()
    print("Generated polynomial_regression_fit.png")
    
    # Run regularization example
    run_regularization_example(base_model)
    print("Generated regularization_effect.png")
    
if __name__ == '__main__':
    main() 