import numpy as np
from tensorflow.keras.models import load_model
from .dataset import load_data

def predict():
    # Load model đã huấn luyện
    model = load_model("and_model.h5")

    X, y = load_data()
    y_pred = model.predict(X)

    print("Predictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]} → Predicted: {y_pred[i][0]:.4f} → Label: {y[i]}")
    return y_pred

if __name__ == "__main__":
    predict()