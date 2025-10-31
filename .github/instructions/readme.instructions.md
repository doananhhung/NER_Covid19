---
applyTo: 'README.md'
---

# Hướng dẫn cho README.md

* Sử dụng cú pháp Markdown chuẩn.
* README phải luôn được cập nhật và bao gồm các phần chính sau:
    1.  **Tiêu đề và Mô tả**: Giới thiệu mục đích dự án (NER, COVID-19, PhoBERT).
    2.  **Entities Recognized**: Một bảng Markdown liệt kê tất cả các nhãn (phải khớp với `src/config.py`).
    3.  **Getting Started**: Hướng dẫn cài đặt, bao gồm `pip install -r requirements.txt`.
    4.  **Usage**: Cung cấp các lệnh chính để chạy:
        * Huấn luyện: `python src/train.py`
        * Đánh giá: `python src/evaluate.py`
        * Chạy Demo: `streamlit run app/app.py`
    5.  **Project Structure**: Sơ đồ cây thư mục giải thích các tệp/thư mục quan trọng.