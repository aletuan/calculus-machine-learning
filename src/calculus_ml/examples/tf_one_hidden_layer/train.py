from .dataset import load_data
from .model_tf import build_model
import numpy as np
import tensorflow as tf
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint
import matplotlib.pyplot as plt
import os

console = Console()

def print_formulas():
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
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/tf_and_training.png')
    plt.close()

def main():
    # Print formulas
    print_formulas()
    
    # Load and print data information
    X, y = load_data()
    console.print("\n[bold]Training Data Information:[/bold]")
    console.print(f"Number of samples: {len(X)}")
    console.print(f"Input features shape: {X.shape}")
    console.print(f"Target values: {y}")
    
    console.print("\n[bold]Model Architecture:[/bold]")
    console.print("- Input Layer: 2 neurons (for x1, x2)")
    console.print("- Hidden Layer: 4 neurons with ReLU activation")
    console.print("- Output Layer: 1 neuron with Sigmoid activation")
    
    print("\n[bold]Training neural network for AND function...[/bold]")
    # Build and train model
    model = build_model()
    history = model.fit(
        X, y,
        epochs=1000,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: console.print(
                    f"Epoch {epoch+1:4d}: loss={logs['loss']:.4f}"
                ) if (epoch + 1) % 100 == 0 else None
            )
        ]
    )
    
    # Plot training history
    plot_training_history(history)
    console.print("\n[bold]Training visualization saved as 'images/tf_and_training.png'[/bold]")
    
    # Evaluate
    loss = model.evaluate(X, y, verbose=0)
    console.print("\n[bold]Final Results:[/bold]")
    console.print(f"Loss: {loss:.4f}")
    
    # Save model
    model.save("src/calculus_ml/examples/tf_one_hidden_layer/and_model.keras")
    console.print("\nModel saved as 'src/calculus_ml/examples/tf_one_hidden_layer/and_model.keras'")
    
    # Make predictions and show results
    y_pred = model.predict(X, verbose=0)
    console.print("\n[bold]Optimal Results:[/bold]")
    console.print(Panel(
        "\n".join([
            "[bold]Predictions:[/bold]",
            "Input → Predicted → Expected",
            "-" * 30,
            *[f"[{X[i][0]}, {X[i][1]}] → {y_pred[i][0]:.4f} → {y[i]}" for i in range(len(X))]
        ]),
        title="Model Predictions",
        border_style="green"
    ))

if __name__ == "__main__":
    main()