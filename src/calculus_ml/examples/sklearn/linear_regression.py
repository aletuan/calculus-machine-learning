"""
Linear Regression example using scikit-learn with Kaggle data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from rich.console import Console
from rich.progress import track
from rich.table import Table
from rich.panel import Panel

# Initialize rich console
console = Console()

def load_data():
    """Load house price data from Kaggle"""
    # Download data from Kaggle
    data_url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    data = pd.read_csv(data_url)
    
    # Select features and target
    X = data[['median_income']].values  # Using median income as feature
    y = data['median_house_value'].values  # House value as target
    
    return X, y

def plot_results(X, y, y_pred, save_path):
    """Plot the results of linear regression"""
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label='Actual Data')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Linear Regression')
    plt.xlabel('Median Income')
    plt.ylabel('Median House Value')
    plt.title('Linear Regression: House Value vs Income')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def print_formulas():
    """Print the mathematical formulas used in linear regression"""
    console.print(Panel(
        "[bold cyan]Linear Regression Formulas[/bold cyan]\n\n"
        "1. Hypothesis Function:\n"
        "   h(x) = θ₀ + θ₁x\n\n"
        "2. Cost Function (Mean Squared Error):\n"
        "   J(θ₀, θ₁) = 1/2m * Σ(h(xⁱ) - yⁱ)²\n\n"
        "3. Gradient Descent:\n"
        "   θⱼ := θⱼ - α * ∂J(θ₀, θ₁)/∂θⱼ\n"
        "   where α is the learning rate",
        border_style="cyan"
    ))

def print_data_info(X, y):
    """Print information about the dataset"""
    table = Table(title="Dataset Information")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Number of samples", str(len(X)))
    table.add_row("Feature range", f"{X.min():.2f} - {X.max():.2f}")
    table.add_row("Target range", f"{y.min():.2f} - {y.max():.2f}")
    table.add_row("Mean feature value", f"{X.mean():.2f}")
    table.add_row("Mean target value", f"{y.mean():.2f}")
    
    console.print(table)

def main():
    """Run the linear regression example"""
    # Ensure images directory exists
    os.makedirs('images', exist_ok=True)
    
    # Print formulas
    print_formulas()
    
    # Load data
    console.print("\n[bold]Loading data...[/bold]")
    X, y = load_data()
    
    # Print data information
    print_data_info(X, y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    console.print("\n[bold]Training the model...[/bold]")
    model = LinearRegression()
    
    # Simulate training progress
    for _ in track(range(100), description="Training progress"):
        model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print model results
    console.print(Panel(
        f"[bold cyan]Model Results[/bold cyan]\n\n"
        f"Intercept (θ₀): {model.intercept_:.2f}\n"
        f"Coefficient (θ₁): {model.coef_[0]:.2f}\n"
        f"Mean Squared Error: {mse:.2f}\n"
        f"R² Score: {r2:.2f}",
        border_style="cyan"
    ))
    
    # Plot and save results
    plot_results(X_test, y_test, y_pred, 'images/sklearn_linear_regression.png')
    console.print("\n[green]Results plot saved to images/sklearn_linear_regression.png[/green]")

if __name__ == "__main__":
    main() 