---
applyTo: '**/*'
---

# Hướng dẫn Kỹ thuật cho Mã nguồn (src/)

## 1. Tiêu chuẩn chung
* Tuân thủ nghiêm ngặt **PEP 8**.
* Sử dụng **Type Hints** (Typing) của Python cho tất cả các tham số hàm và giá trị trả về (ví dụ: `def my_func(name: str) -> int:`).
* Viết **Docstrings** (theo kiểu Google hoặc NumPy) cho tất cả các class và hàm public.
* Sử dụng tên biến và hàm có ý nghĩa, rõ ràng, tránh viết tắt không cần thiết.
* Trả lời các ngoại lệ (exceptions) một cách cụ thể, tránh sử dụng `except:` chung chung.
* Trả lời tôi bằng tiếng Việt.
* Không sử dụng icon cảm xúc (emoji) trong code hoặc comment.
* Không tạo ra các file .md mới.

## 2. Phong cách Code (Quan trọng)
* **Hạn chế `print`**: Tránh sử dụng câu lệnh `print()` cho việc gỡ lỗi (debug), hạn chế tốt đa việc dùng `print()` để thông báo bắt đầu làm gì đó. Ưu tiên sử dụng mô-đun `logging` nếu cần.
* **Comment**: Viết comment ngắn gọn, rõ ràng và đầy đủ. Comment nên giải thích logic phức tạp hoặc "tại sao" (why) một đoạn code tồn tại.

## 3. Quy tắc của Dự án (Bắt buộc)

### 3.1. Cấu hình (Config)
* **KHÔNG BAO GIỜ** hardcode đường dẫn tệp, tên mô hình, hoặc siêu tham số (batch size, learning rate) trực tiếp trong script.
* **LUÔN LUÔN** import và sử dụng các hằng số từ `src.config` (ví dụ: `config.MAX_LEN`, `config.PRE_TRAINED_MODEL_NAME`).

### 3.2. PyTorch & Transformers
* **Model**: Sử dụng `AutoModelForTokenClassification` và `AutoTokenizer`.
* **Device**: Luôn kiểm tra và sử dụng GPU nếu có (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`) và chuyển model/tensor lên device.
* **Training Loop (`train.py`)**:
    * Gọi `model.train()` khi huấn luyện và `model.eval()` khi đánh giá.
    * Bọc phần đánh giá trong `with torch.no_grad():`.
* **Dataset (`dataset.py`)**:
    * Logic `__getitem__` phải xử lý chính xác việc **căn chỉnh nhãn (label alignment)**.
    * Gán `config.SUBWORD_TAG_ID` (-100) cho tất cả các sub-word token *không phải* là token đầu tiên của một từ.

### 3.3. Inference (`inference.py`)
* Phải sử dụng `VnCoreNLP` để **tách từ** văn bản đầu vào trước khi token hóa.