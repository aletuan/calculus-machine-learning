import numpy as np
from typing import List, Tuple, Dict, Optional

class DecisionNode:
    """Node trong cây quyết định"""
    def __init__(self, feature_index: Optional[int] = None,
                 threshold: Optional[float] = None,
                 left: Optional['DecisionNode'] = None,
                 right: Optional['DecisionNode'] = None,
                 value: Optional[int] = None):
        self.feature_index = feature_index  # Index của feature để split
        self.threshold = threshold          # Ngưỡng để split
        self.left = left                    # Node con trái
        self.right = right                  # Node con phải
        self.value = value                  # Giá trị dự đoán (cho leaf node)

class DecisionTree:
    """
    Cài đặt Decision Tree sử dụng Information Gain và Entropy
    
    Thuộc tính:
    -----------
    max_depth : int
        Độ sâu tối đa của cây
    min_samples_split : int
        Số lượng mẫu tối thiểu để tiếp tục split
    root : DecisionNode
        Node gốc của cây
        
    Phương thức:
    -----------
    fit(X, y) : Huấn luyện cây quyết định
    predict(X) : Dự đoán nhãn cho dữ liệu mới
    """
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def _entropy(self, y: np.ndarray) -> float:
        """
        Tính entropy của tập dữ liệu
        
        H(S) = -Σ p(x) * log2(p(x))
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _information_gain(self, y: np.ndarray, y_left: np.ndarray, 
                         y_right: np.ndarray) -> float:
        """
        Tính information gain khi split dữ liệu
        
        IG(S, A) = H(S) - Σ |Sv|/|S| * H(Sv)
        """
        parent_entropy = self._entropy(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        child_entropy = (n_left/n) * self._entropy(y_left) + \
                       (n_right/n) * self._entropy(y_right)
        return parent_entropy - child_entropy
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        """
        Tìm split tốt nhất dựa trên information gain
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> DecisionNode:
        """
        Xây dựng cây quyết định theo thuật toán đệ quy
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Điều kiện dừng
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            leaf_value = np.argmax(np.bincount(y))
            return DecisionNode(value=leaf_value)
        
        # Tìm split tốt nhất
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            leaf_value = np.argmax(np.bincount(y))
            return DecisionNode(value=leaf_value)
        
        # Tạo các node con
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Huấn luyện cây quyết định"""
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    def _traverse_tree(self, x: np.ndarray, node: DecisionNode) -> int:
        """Duyệt cây để dự đoán"""
        if node.value is not None:
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán nhãn cho dữ liệu mới"""
        return np.array([self._traverse_tree(x, self.root) for x in X]) 