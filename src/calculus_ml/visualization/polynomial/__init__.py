"""
Visualization utilities for polynomial regression.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_polynomial_comparison(X, y, models, degrees, save_path=None):
    """Plot comparison of polynomial regressions with different degrees.
    
    Parameters
    ----------
    X : array-like
        Input features
    y : array-like
        Target values
    models : list of PolynomialRegression
        Fitted models with different degrees
    degrees : list of int
        Degrees of polynomials
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    plt.scatter(X, y, color='black', alpha=0.5, label='Training Data')
    
    # Sort X for smooth curve plotting
    X_sort = np.sort(X)
    
    # Plot predictions for each model
    colors = ['green', 'orange', 'red']
    for model, degree, color in zip(models, degrees, colors):
        y_pred = model.predict(X_sort)
        plt.plot(X_sort, y_pred, color=color, 
                label=f'Degree {degree}', linewidth=2)
    
    plt.xlabel('Diện tích (1000 sqft)')
    plt.ylabel('Giá nhà (1000$)')
    plt.title('So sánh các mô hình Polynomial Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_regularization_effect(X, y, models, lambdas, save_path=None):
    """Plot the effect of regularization on polynomial regression.
    
    Parameters
    ----------
    X : array-like
        Input features
    y : array-like
        Target values
    models : list of PolynomialRegression
        Fitted models with different lambda values
    lambdas : list of float
        Lambda values used for regularization
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    plt.scatter(X, y, color='black', alpha=0.5, label='Training Data')
    
    # Sort X for smooth curve plotting
    X_sort = np.sort(X)
    
    # Plot predictions for each model
    colors = ['blue', 'orange', 'red']
    labels = ['λ nhỏ (Overfitting)', 'λ tối ưu', 'λ lớn (Underfitting)']
    
    for model, lambda_val, color, label in zip(models, lambdas, colors, labels):
        y_pred = model.predict(X_sort)
        plt.plot(X_sort, y_pred, color=color, 
                label=f'{label} (λ={lambda_val})', linewidth=2)
        
        # Add confidence band (just for visualization)
        if color == 'orange':  # Only for optimal model
            std = np.std(y - model.predict(X))
            plt.fill_between(X_sort, 
                           y_pred - 2*std, 
                           y_pred + 2*std,
                           color='gray', alpha=0.2,
                           label='95% Confidence Interval')
    
    plt.xlabel('Diện tích (1000 sqft)')
    plt.ylabel('Giá nhà (1000$)')
    plt.title('Ảnh hưởng của Regularization trong Polynomial Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 