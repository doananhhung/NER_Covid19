---
applyTo: 'notebooks/**/*.ipynb'
---

# Hướng dẫn cho Jupyter Notebook

## 1. Tổ chức Notebook

### Cấu trúc
1.  **Tiêu đề & Mô tả**: Cell Markdown đầu tiên phải chứa Tiêu đề (cấp 1) và mô tả ngắn gọn mục đích của notebook.
2.  **Imports**: Tất cả các lệnh import phải được gom lại trong 1-2 cell code đầu tiên.
3.  **Cấu hình**: Khai báo các hằng số và biến cấu hình (ví dụ: đường dẫn, seed). Nếu có thể, hãy import từ `src.config`.
4.  **Nội dung chính**: Các bước tải dữ liệu, phân tích (EDA), xử lý, và code logic.
5.  **Kết luận**: Cell Markdown cuối cùng tóm tắt các phát hiện hoặc kết quả.

### Code Cells
* Mỗi cell nên tập trung vào một mục đích rõ ràng (ví dụ: tải dữ liệu, visualize, định nghĩa hàm).
* Không viết cell code quá dài (tối đa 30-40 dòng).
* Luôn thêm một cell Markdown phía trước để giải thích cell code sắp thực hiện làm gì.
* Hiển thị các output quan trọng (biểu đồ, bảng thống kê, metrics).

## 2. Tiêu chuẩn Code
* Sử dụng `pandas` để đọc và thao tác dữ liệu.
* Sử dụng `matplotlib` hoặc `seaborn` cho visualization.
* Khi phân tích dữ liệu, hãy tham chiếu đến các tệp trong `data/raw/PhoNER_COVID19/` (ví dụ: `train_word.json`).