---
applyTo: 'app/**/*.py'
---

# Hướng dẫn cho Ứng dụng Streamlit (app/)

## 1. Tiêu chuẩn Streamlit
* Sử dụng các hàm chuẩn của Streamlit: `st.title`, `st.markdown`, `st.text_area`, `st.button`, `st.spinner`, `st.expander`.
* Sử dụng `st.cache_resource` để tải mô hình `NERPredictor`. Điều này đảm bảo mô hình chỉ được tải một lần duy nhất khi khởi động ứng dụng.
* Sử dụng `st.error`, `st.warning`, `st.success` để hiển thị thông báo trạng thái.
* Sử dụng `st.markdown(..., unsafe_allow_html=True)` để hiển thị văn bản đã được tô màu (highlight) thực thể.

## 2. Tương tác với Dự án
* Để tải mô hình, import lớp `NERPredictor` từ `src.inference`.
* Để hiển thị kết quả, import hàm `render_entities` từ `app.utils`.
* Phải xử lý `sys.path` để thêm thư mục gốc của dự án, cho phép import các mô-đun từ `src/` (vì `app/` và `src/` là anh em).

## 3. Logic Ứng dụng
* Lấy văn bản đầu vào từ `st.text_area`.
* Khi nhấn `st.button`, gọi hàm `predictor.predict(text)` bên trong `st.spinner`.
* Truyền kết quả (danh sách entities) và văn bản gốc cho hàm `render_entities` để hiển thị.
* Cung cấp tùy chọn xem kết quả dạng JSON thô bằng `st.json()`.