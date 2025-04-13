"""
Train neural network for AND function using TensorFlow.
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

def print_formulas():
    """Print neural network formulas."""
    console.print(Panel("""[bold]Neural Network Formulas:[/bold]

1. Forward Propagation:
   - Hidden Layer: h = ReLU(W1·X + b1)
   - Output Layer: ŷ = σ(W2·h + b2)
   where σ(z) = 1/(1 + e^(-z)) is the sigmoid function

2. Loss Function (Binary Cross Entropy):
   L(y, ŷ) = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

3. Optimization:
   - Using Adam optimizer with learning rate = 0.01
   - Weights updated using gradients computed through backpropagation""",
    title="Formulas", border_style="blue"))

def plot_training_history(history):
    """Plot and save training history."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'])
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('images/tf_and_training.png')
    plt.close()

def main():
    # Print formulas
    print_formulas()
    
    # Training data for AND function
    X = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
    y = tf.constant([[0], [0], [0], [1]], dtype=tf.float32)
    
    # Print data information
    console.print("\n[bold]Training Data Information:[/bold]")
    console.print(f"Number of samples: {len(X)}")
    console.print(f"Input features shape: {X.shape}")
    console.print(f"Target values: {y.numpy().flatten().tolist()}")
    
    # Build model
    from .model_tf import build_model
    model = build_model()
    
    # Print model architecture
    console.print("\n[bold]Model Architecture:[/bold]")
    console.print("- Input Layer: 2 neurons (for x1, x2)")
    console.print("- Hidden Layer: 4 neurons with ReLU activation")
    console.print("- Output Layer: 1 neuron with Sigmoid activation")
    
    # Display training process
    console.print("\n[bold yellow]Training Process:[/bold yellow]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Training...", total=1000)
        
        class TrainingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress.update(task, advance=1)
                if epoch % 100 == 0 or epoch == 999:
                    console.print(f"Epoch {epoch:4d}: Loss = {logs['loss']:.4f}")
        
        # Train model
        history = model.fit(
            X, y,
            epochs=1000,
            verbose=0,
            callbacks=[TrainingCallback()]
        )
    
    # Plot training history
    plot_training_history(history)
    console.print("\nTraining visualization saved as 'images/tf_and_training.png'")
    
    # Save model
    model.save("src/calculus_ml/examples/tf_one_hidden_layer/and_model.keras")
    console.print("\nModel saved as 'src/calculus_ml/examples/tf_one_hidden_layer/and_model.keras'")
    
    # Make predictions and show results
    predictions = model.predict(X, verbose=0)
    
    # Create results panel
    results = [
        "[bold green]Optimal Results:[/bold green]",
        "Input → Predicted → Expected",
        "-" * 35
    ]
    
    for i, (input_data, pred) in enumerate(zip(X, predictions)):
        expected = 1 if all(x == 1 for x in input_data.numpy()) else 0
        results.append(
            f"{input_data.numpy().tolist()} → {pred[0]:.4f} → {expected}"
        )
    
    # Display results in panel
    console.print(Panel(
        "\n".join(results),
        title="Model Predictions",
        border_style="cyan"
    ))

if __name__ == "__main__":
    main()