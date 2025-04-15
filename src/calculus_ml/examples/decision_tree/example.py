import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ...core.decision_tree.tree import DecisionTree

def generate_data():
    """
    Tạo dữ liệu mẫu cho bài toán phân loại
    Dataset: Phân loại hoa iris (đơn giản hóa thành 2 class)
    """
    # Load dữ liệu iris và chỉ lấy 2 class đầu tiên
    iris = load_iris()
    X = iris.data[:100, [0, 1]]  # Chỉ lấy 2 features đầu tiên
    y = iris.target[:100]        # Chỉ lấy 2 classes đầu tiên
    
    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def plot_decision_boundary(X, y, model):
    """
    Vẽ decision boundary của model
    """
    h = 0.02  # Kích thước grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Dự đoán cho toàn bộ grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Vẽ contour
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.colorbar()

def run_decision_tree_example():
    """
    Ví dụ về Decision Tree với dataset Iris
    """
    print("\nDecision Tree Example")
    print("====================")
    
    # 1. Tạo dữ liệu
    X_train, X_test, y_train, y_test = generate_data()
    print("\nDữ liệu huấn luyện:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    # 2. Khởi tạo và huấn luyện mô hình
    tree = DecisionTree(max_depth=3, min_samples_split=2)
    tree.fit(X_train, y_train)
    
    # 3. Dự đoán và đánh giá
    y_pred = tree.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"\nĐộ chính xác trên tập test: {accuracy:.2f}")
    
    # 4. Trực quan hóa
    plt.figure(figsize=(10, 5))
    
    # 4.1. Vẽ dữ liệu huấn luyện và decision boundary
    plt.subplot(121)
    plot_decision_boundary(X_train, y_train, tree)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
    plt.title('Decision Boundary trên tập train')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    
    # 4.2. Vẽ dữ liệu test và dự đoán
    plt.subplot(122)
    plot_decision_boundary(X_test, y_test, tree)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')
    plt.title('Dự đoán trên tập test')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    
    plt.tight_layout()
    plt.savefig('images/decision_tree_boundary.png')
    plt.close()

if __name__ == "__main__":
    run_decision_tree_example() 