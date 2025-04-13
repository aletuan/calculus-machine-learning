from .dataset import load_data
from .model_tf import build_model
import numpy as np

def main():
    # Load data
    X, y = load_data()
    
    print("\nTraining neural network for AND function...")
    # Build and train model
    model = build_model()
    history = model.fit(X, y, epochs=1000, verbose=1)
    
    # Evaluate
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"\nFinal Results:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    
    # Save model
    model.save("and_model.h5")
    print("\nModel saved as 'and_model.h5'")
    
    # Make predictions
    y_pred = model.predict(X, verbose=0)
    print("\nPredictions:")
    print("Input → Predicted → Expected")
    print("-" * 30)
    for i in range(len(X)):
        print(f"[{X[i][0]}, {X[i][1]}] → {y_pred[i][0]:.4f} → {y[i]}")

if __name__ == "__main__":
    main()