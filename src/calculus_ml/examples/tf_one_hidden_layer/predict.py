import tensorflow as tf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

def predict_and_function():
    # Load the trained model
    model = tf.keras.models.load_model("src/calculus_ml/examples/tf_one_hidden_layer/and_model.keras")
    
    # Test data for AND function
    test_data = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    
    # Make predictions
    predictions = model.predict(test_data)
    
    # Create a table to display results
    table = Table(title="AND Function Predictions")
    table.add_column("Input", style="cyan")
    table.add_column("Predicted", style="green")
    table.add_column("Expected", style="yellow")
    
    # Add rows to table
    for i, (input_data, pred) in enumerate(zip(test_data, predictions)):
        expected = 1 if input_data == [1, 1] else 0
        table.add_row(
            str(input_data),
            f"{pred[0]:.4f}",
            str(expected)
        )
    
    # Display results
    console = Console()
    console.print(Panel(table, title="Neural Network Predictions", border_style="blue"))

if __name__ == "__main__":
    predict_and_function()