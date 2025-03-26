# Calculus and Machine Learning Examples

A collection of examples demonstrating vector operations and their applications in machine learning.

## Project Structure

```
calculus-machine-learning/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   └── calculus_ml/
│       ├── __init__.py
│       ├── vector_operations.py
│       ├── vector_visualization.py
│       ├── linear_regression.py
│       └── main.py
└── images/
    └── ...
```

## Features

1. **Vector cơ bản và biểu diễn trực quan**
   - Tạo và vẽ vector trong không gian 2D
   - Minh họa phép cộng và trừ vector

2. **Phép toán vector cơ bản**
   - Cộng, trừ vector
   - Nhân vector với số vô hướng
   - Tính độ dài (norm) của vector
   - Tích vô hướng (dot product)
   - Tích có hướng (cross product)

3. **Vector đơn vị và góc giữa hai vector**
   - Chuẩn hóa vector thành vector đơn vị
   - Tính góc giữa hai vector
   - Biểu diễn trực quan góc giữa vector

4. **Ứng dụng trong Linear Regression**
   - Tạo dữ liệu đơn giản
   - Sử dụng phép tính vector để tìm hệ số hồi quy
   - Vẽ đường hồi quy

5. **Chiếu vector và ứng dụng**
   - Chiếu vector lên một vector khác
   - Biểu diễn trực quan phép chiếu

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aletuan/calculus-machine-learning.git
cd calculus-machine-learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

Run the main script to see all examples:
```bash
python -m calculus_ml.main
```

## Development

To contribute to the project:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.