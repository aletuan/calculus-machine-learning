"""
Multiple Linear Regression example using scikit-learn with California Housing dataset.
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
    """Load California housing data from Kaggle"""
    # Download data from Kaggle
    data_url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    data = pd.read_csv(data_url)
    
    # Select numerical features only
    features = [
        'median_income',        # Thu nhập trung bình
        'housing_median_age',   # Tuổi trung bình của nhà
        'total_rooms',          # Tổng số phòng
        'total_bedrooms',       # Tổng số phòng ngủ
        'population',           # Dân số
        'households',           # Số hộ gia đình
        'longitude',            # Kinh độ
        'latitude'              # Vĩ độ
    ]
    
    # Fill missing values with median for numerical features
    for feature in features:
        data[feature] = data[feature].fillna(data[feature].median())
    
    X = data[features].values
    y = data['median_house_value'].values
    
    return X, y, features

def plot_feature_importance(model, features, save_path):
    """Plot feature importance based on coefficients"""
    plt.figure(figsize=(12, 6))
    coefficients = model.coef_
    importance = np.abs(coefficients)
    importance = importance / importance.sum() * 100  # Convert to percentage
    
    plt.bar(features, importance)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Feature Importance (%)')
    plt.title('Feature Importance in Multiple Linear Regression')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def print_formulas():
    """Print the mathematical formulas used in multiple linear regression"""
    console.print(Panel(
        "[bold cyan]Multiple Linear Regression Formulas[/bold cyan]\n\n"
        "1. Hypothesis Function:\n"
        "   h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ\n\n"
        "2. Cost Function (Mean Squared Error):\n"
        "   J(θ) = 1/2m * Σ(h(xⁱ) - yⁱ)²\n\n"
        "3. Gradient Descent:\n"
        "   θⱼ := θⱼ - α * ∂J(θ)/∂θⱼ\n"
        "   where α is the learning rate",
        border_style="cyan"
    ))

def print_data_info(X, y, features):
    """Print information about the dataset"""
    table = Table(title="Dataset Information")
    table.add_column("Feature", style="cyan")
    table.add_column("Min", style="green")
    table.add_column("Max", style="green")
    table.add_column("Mean", style="green")
    
    for i, feature in enumerate(features):
        table.add_row(
            feature,
            f"{X[:, i].min():.2f}",
            f"{X[:, i].max():.2f}",
            f"{X[:, i].mean():.2f}"
        )
    
    table.add_row(
        "Target (median_house_value)",
        f"{y.min():.2f}",
        f"{y.max():.2f}",
        f"{y.mean():.2f}"
    )
    
    console.print(table)

def main():
    """Run the multiple linear regression example"""
    # Ensure images directory exists
    os.makedirs('images', exist_ok=True)
    
    # Print formulas
    print_formulas()
    
    # Load data
    console.print("\n[bold]Loading data...[/bold]")
    X, y, features = load_data()
    
    # Print data information
    print_data_info(X, y, features)
    
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
        f"R² Score: {r2:.2f}\n"
        f"Mean Squared Error: {mse:.2f}",
        border_style="cyan"
    ))
    
    # Print feature coefficients
    table = Table(title="Feature Coefficients")
    table.add_column("Feature", style="cyan")
    table.add_column("Coefficient", style="green")
    
    for feature, coef in zip(features, model.coef_):
        table.add_row(feature, f"{coef:.2f}")
    
    console.print(table)
    
    # Plot and save feature importance
    plot_feature_importance(model, features, 'images/sklearn_multiple_linear_regression.png')
    console.print("\n[green]Feature importance plot saved to images/sklearn_multiple_linear_regression.png[/green]")

if __name__ == "__main__":
    main() 