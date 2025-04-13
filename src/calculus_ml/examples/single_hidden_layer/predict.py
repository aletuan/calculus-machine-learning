from data_and import load_data
from model_numpy import OneHiddenLayerNN

X, y = load_data()
model = OneHiddenLayerNN()
# Gán lại trọng số nếu cần load lại từ file – hoặc train trước rồi predict

# Đoán sau khi đã huấn luyện (nếu chạy liên tục)
y_hat = model.forward(X)
predictions = (y_hat > 0.5).astype(int)

print("Input:")
print(X)
print("Predicted Output:")
print(predictions)
print("Ground Truth:")
print(y)