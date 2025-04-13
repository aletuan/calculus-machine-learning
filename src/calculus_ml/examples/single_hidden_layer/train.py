"""
Huấn luyện mạng neural với dữ liệu XOR.
"""

import numpy as np
from .model_numpy import NeuralNetwork
from .activations import sigmoid, sigmoid_derivative

def main():
    # Dữ liệu XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Khởi tạo mạng neural
    input_size = 2
    hidden_size = 4
    output_size = 1
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    
    # Huấn luyện mạng
    nn.train(X, y, epochs=10000)
    
    # Dự đoán và in kết quả
    predictions = nn.predict(X)
    print("\nDự đoán:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y[i]}, Prediction: {predictions[i]}")
    
if __name__ == "__main__":
    main()