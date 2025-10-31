---
applyTo: 'src/**/*.py'
---

# Hướng dẫn Kỹ thuật cho Mã nguồn (src/)

## 1. Tiêu chuẩn chung
* Tuân thủ nghiêm ngặt **PEP 8**.
* Sử dụng **Type Hints** (Typing) của Python cho tất cả các tham số hàm và giá trị trả về (ví dụ: `def my_func(name: str) -> int:`).
* Viết **Docstrings** (theo kiểu Google hoặc NumPy) cho tất cả các class và hàm public.

## 2. Quy tắc của Dự án (Bắt buộc)

### 2.1. Cấu hình (Config)
* **KHÔNG BAO GIỜ** hardcode đường dẫn tệp, tên mô hình, hoặc siêu tham số (batch size, learning rate) trực tiếp trong script.
* **LUÔN LUÔN** import và sử dụng các hằng số từ `src.config` (ví dụ: `config.MAX_LEN`, `config.PRE_TRAINED_MODEL_NAME`, `config.MODEL_OUTPUT_DIR`).

### 2.2. PyTorch & Transformers
* **Model**: Sử dụng `AutoModelForTokenClassification` và `AutoTokenizer` từ thư viện `transformers` để tải `vinai/phobert-base`.
* **Device**: Luôn kiểm tra và sử dụng GPU nếu có:
    `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
    Phải chuyển (`.to(device)`) tất cả model và các batch tensor lên device này.
* **Optimizer**: Ưu tiên sử dụng `AdamW` (từ `torch.optim` hoặc `transformers`).
* **Scheduler**: Sử dụng `get_linear_schedule_with_warmup` từ `transformers`.
* **Training Loop (`train.py`)**:
    * Luôn gọi `model.train()` khi huấn luyện và `model.eval()` khi đánh giá.
    * Vòng lặp phải bao gồm: `optimizer.zero_grad()`, `loss.backward()`, `torch.nn.utils.clip_grad_norm_`, và `optimizer.step()`.
* **Evaluation (`evaluate.py`)**:
    * Phải được bọc trong `with torch.no_grad():`.
    * Sử dụng `seqeval` để tính toán F1, Precision, Recall.
    * Khi chuẩn bị dữ liệu cho `seqeval`, phải loại bỏ tất cả các dự đoán (predictions) và nhãn (labels) có ID là `config.SUBWORD_TAG_ID` (-100) trước khi so sánh.

### 2.3. Dataset (`dataset.py`)
* Khi định nghĩa `NerDataset`, logic `__getitem__` phải xử lý chính xác việc **căn chỉnh nhãn (label alignment)**.
* Token đầu tiên của một từ (word) sẽ nhận nhãn thực thể (ví dụ: `B-LOCATION`).
* Tất cả các sub-word token tiếp theo của từ đó *phải* nhận nhãn `config.SUBWORD_TAG_ID` (-100) để được hàm loss bỏ qua.

### 2.4. Inference (`inference.py`)
* Lớp `NERPredictor` phải sử dụng `VnCoreNLP` (từ `py_vncorenlp`) để **tách từ (word segmentation)** cho văn bản đầu vào *trước khi* đưa vào tokenizer của PhoBERT.
* Logic hậu kỳ (post-processing) phải nhóm các BPE token (`@@`) và các nhãn `B-` / `I-` liên tiếp lại thành các thực thể (entity) hoàn chỉnh.