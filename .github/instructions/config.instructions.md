---
applyTo: 'src/config.py'
---

# Hướng dẫn cho file config.py

* Tất cả các hằng số phải được đặt tên theo kiểu **UPPER_SNAKE_CASE** (ví dụ: `MAX_LEN`).
* Sử dụng `os.path.join` và `os.path.dirname` để xây dựng đường dẫn tệp một cách an toàn, thay vì nối chuỗi.
* Các hằng số phải được nhóm rõ ràng thành các phần:
    1.  Paths (Đường dẫn)
    2.  Model Configuration (Cấu hình Model)
    3.  Training Hyperparameters (Siêu tham số Huấn luyện)
    4.  Tag Configuration (Cấu hình Nhãn)
* Khi cập nhật `UNIQUE_TAGS`, phải đảm bảo `TAGS_TO_IDS` và `IDS_TO_TAGS` được cập nhật tương ứng.
* `SUBWORD_TAG_ID` phải luôn là `-100`.