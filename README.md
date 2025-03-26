# Ứng Dụng Giải Tích và Học Máy

Bộ ví dụ minh họa về các phép toán vector và ứng dụng trong học máy.

## Cấu Trúc Dự Án

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

## Tính Năng

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

## Hướng Dẫn Cài Đặt

1. Tải mã nguồn:
```bash
git clone https://github.com/aletuan/calculus-machine-learning.git
cd calculus-machine-learning
```

2. Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

3. Cài đặt gói phần mềm ở chế độ phát triển:
```bash
pip install -e .
```

## Cách Sử Dụng

Chạy chương trình chính để xem tất cả các ví dụ:
```bash
python -m calculus_ml.main
```

## Phát Triển

Để đóng góp cho dự án:

1. Fork repository này
2. Tạo nhánh mới cho tính năng của bạn
3. Thực hiện các thay đổi
4. Gửi pull request

## Giấy Phép

Dự án này được cấp phép theo giấy phép MIT - xem file LICENSE để biết thêm chi tiết.