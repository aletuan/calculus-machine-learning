"""
Train a perceptron to learn the AND gate function.
This example demonstrates how a simple neural network can learn basic logical functions.
"""

from .perceptron import Perceptron
from .and_dataset import load_and_data

def train_perceptron():
    """Train a perceptron to learn the AND gate function"""
    # Load the AND gate dataset
    X, y = load_and_data()

    # Initialize and train the perceptron
    model = Perceptron(input_dim=2, lr=0.1)  # 2 input features, learning rate 0.1
    model.train(X, y, epochs=1000)           # Train for 1000 epochs

    # Test the model on all possible inputs
    print("\nTesting the trained perceptron:")
    print("Input (x1, x2) | Prediction | Label")
    print("-" * 35)
    for x_i, y_i in zip(X, y):
        y_hat = model.predict(x_i)
        print(f"({x_i[0]}, {x_i[1]})        | {y_hat:.4f}    | {y_i}")

if __name__ == "__main__":
    train_perceptron()