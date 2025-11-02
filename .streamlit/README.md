# Cấu hình Streamlit

Thư mục này chứa các file cấu hình cho ứng dụng Streamlit.

## Files

### `config.toml`
Cấu hình giao diện và server của Streamlit (đã được commit).

### `secrets.toml` (KHÔNG commit)
File chứa API keys và thông tin nhạy cảm - **ĐÃ ĐƯỢC THÊM VÀO .gitignore**.

## Cách thiết lập secrets.toml

### Bước 1: Tạo file từ template

```bash
# Windows Command Prompt
copy secrets.toml.example secrets.toml

# Windows PowerShell
Copy-Item secrets.toml.example secrets.toml

# Linux/Mac
cp secrets.toml.example secrets.toml
```

### Bước 2: Lấy Gemini API Key

1. Truy cập: https://makersuite.google.com/app/apikey
2. Đăng nhập với tài khoản Google
3. Nhấn "Create API Key"
4. Sao chép API key

### Bước 3: Cập nhật secrets.toml

Mở file `secrets.toml` và thay thế:

```toml
[gemini]
api_key = "your-gemini-api-key-here"
```

Thành:

```toml
[gemini]
api_key = "AIzaSy..."  # API key thực của bạn
```

### Bước 4: Xác nhận file không bị commit

Chạy lệnh:

```bash
git status
```

File `secrets.toml` **KHÔNG được** xuất hiện trong danh sách "Untracked files" hoặc "Changes to be committed".

Nếu xuất hiện, kiểm tra lại file `.gitignore` ở thư mục gốc.

## Bảo mật

⚠️ **QUAN TRỌNG:**

- **KHÔNG BAO GIỜ** commit file `secrets.toml` lên Git
- **KHÔNG BAO GIỜ** chia sẻ API key của bạn
- Nếu vô tình commit API key, hãy:
  1. Xóa API key cũ trên Google AI Studio
  2. Tạo API key mới
  3. Xóa file khỏi Git history (sử dụng `git filter-branch` hoặc BFG Repo-Cleaner)

## Sử dụng secrets trong code

```python
import streamlit as st

# Truy cập API key
api_key = st.secrets["gemini"]["api_key"]
```

Nếu file `secrets.toml` không tồn tại, Streamlit sẽ báo lỗi khi truy cập `st.secrets`.
