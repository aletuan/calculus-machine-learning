"""
Main script to run all examples.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def run_linear_example():
    """Run linear regression example."""
    from .linear_regression.single_linear_example import run_linear_example
    run_linear_example()

def run_perceptron_example():
    """Run perceptron example."""
    from .perceptron.train import train_perceptron
    train_perceptron()

def run_single_hidden_layer_example():
    """Run single hidden layer example."""
    from .single_hidden_layer.train import main
    main()

def run_polynomial_example():
    """Run polynomial regression example."""
    from .polynomial_example import main
    main()

def run_tf_one_hidden_layer_example():
    """Run TensorFlow one hidden layer example."""
    from .tf_one_hidden_layer.train import main
    main()

def main():
    """Run all examples."""
    # Create table for examples
    table = Table(title="Machine Learning Examples")
    table.add_column("Example", style="cyan")
    table.add_column("Description", style="green")
    
    # Add examples to table
    table.add_row(
        "1. Linear Regression",
        "Single variable linear regression with gradient descent"
    )
    table.add_row(
        "2. Perceptron",
        "Perceptron learning AND function"
    )
    table.add_row(
        "3. Single Hidden Layer",
        "Neural network with one hidden layer learning XOR function"
    )
    table.add_row(
        "4. Polynomial Regression",
        "Polynomial regression with different degrees and regularization"
    )
    table.add_row(
        "5. TensorFlow One Hidden Layer",
        "Neural network using TensorFlow learning AND function"
    )
    
    # Display table
    console.print(Panel(table, title="Available Examples", border_style="blue"))
    
    # Get user choice
    while True:
        try:
            choice = int(console.input("\n[bold]Enter example number to run (1-5): [/bold]"))
            if 1 <= choice <= 5:
                break
            console.print("[red]Please enter a number between 1 and 5[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")
    
    # Run selected example
    console.print("\n[bold yellow]Running example...[/bold yellow]")
    if choice == 1:
        run_linear_example()
    elif choice == 2:
        run_perceptron_example()
    elif choice == 3:
        run_single_hidden_layer_example()
    elif choice == 4:
        run_polynomial_example()
    elif choice == 5:
        run_tf_one_hidden_layer_example()

if __name__ == "__main__":
    main() 