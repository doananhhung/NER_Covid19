# Vietnamese COVID-19 Named Entity Recognition (NER)

Dự án **Named Entity Recognition (NER)** cho văn bản tiếng Việt liên quan đến COVID-19, sử dụng mô hình **PhoBERT** để nhận diện và trích xuất thông tin bệnh nhân từ các văn bản y tế.

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Entities được nhận diện](#entities-được-nhận-diện)
- [Tính năng chính](#tính-năng-chính)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Dataset](#dataset)
- [Mô hình](#mô-hình)

---

## Giới thiệu

Dự án này xây dựng một hệ thống NER (Named Entity Recognition) để tự động nhận diện và trích xuất thông tin từ các văn bản y tế tiếng Việt liên quan đến COVID-19. Hệ thống sử dụng:

- **PhoBERT** (`vinai/phobert-base`) - Mô hình ngôn ngữ tiếng Việt pre-trained
- **VnCoreNLP** - Công cụ tách từ tiếng Việt
- **PhoNER_COVID19** - Dataset được gán nhãn cho bài toán NER
- **Streamlit** - Giao diện web demo tương tác
- **Gemini AI** - Hỗ trợ trích xuất thông tin tự động (chế độ Auto)

### Ứng dụng thực tế

- Trích xuất thông tin bệnh nhân từ báo cáo y tế
- Tự động hóa việc ghi nhận thông tin trong hệ thống quản lý y tế
- Hỗ trợ phân tích dữ liệu dịch bệnh COVID-19

---

## Entities được nhận diện

Hệ thống nhận diện **10 loại entities** chính theo định dạng BIO tagging:

| Entity Type | Mô tả | Ví dụ |
|-------------|-------|-------|
| **PATIENT_ID** | Mã số bệnh nhân | BN123, Bệnh nhân 456 |
| **NAME** | Họ và tên bệnh nhân | Nguyễn Văn A, Trần Thị B |
| **AGE** | Tuổi, độ tuổi | 35 tuổi, 40 |
| **GENDER** | Giới tính | nam, nữ |
| **JOB** | Nghề nghiệp | bác sĩ, công nhân |
| **LOCATION** | Địa điểm | Hà Nội, quận 1, phường Bến Nghé |
| **ORGANIZATION** | Tổ chức, cơ quan | Bệnh viện Bạch Mai, CDC |
| **SYMPTOM_AND_DISEASE** | Triệu chứng và bệnh | sốt, ho, COVID-19 |
| **TRANSPORTATION** | Phương tiện di chuyển | xe buýt, chuyến bay VN123 |
| **DATE** | Ngày tháng, thời gian | 15/3/2021, ngày 20 tháng 4 |

**Định dạng tagging:**
- `B-[ENTITY]`: Beginning - Token đầu tiên của entity
- `I-[ENTITY]`: Inside - Token tiếp theo của entity
- `O`: Outside - Không thuộc entity nào

---

## Tính năng chính

### 1. Training & Evaluation
- Huấn luyện mô hình NER với PhoBERT
- Đánh giá mô hình với metrics (Precision, Recall, F1-score)
- Hỗ trợ fine-tuning với custom hyperparameters

### 2. Inference
- Dự đoán entities từ văn bản tiếng Việt
- Tự động tách từ với VnCoreNLP
- Xử lý chính xác sub-word tokens

### 3. Web Application (Streamlit)
Ứng dụng web với 2 chế độ hoạt động:

#### **Manual Mode (Chế độ Thủ công)**
- Nhập văn bản trực tiếp
- Hiển thị entities được nhận diện với highlight màu sắc
- Trích xuất thông tin bệnh nhân từ văn bản đơn

#### **Auto Mode (Chế độ Tự động)**
- Tích hợp Gemini AI để tự động tách văn bản nhiều bệnh nhân
- Trích xuất thông tin nhiều bệnh nhân cùng lúc
- Xuất kết quả dưới dạng JSON hoặc CSV
- Phù hợp xử lý văn bản dài, phức tạp

### 4. Chrome Extension (MỚI)
Extension trình duyệt để sử dụng NER trực tiếp trên web:

#### **Tính năng chính:**
- Xử lý văn bản từ trang web hiện tại hoặc nhập thủ công
- Highlight entities trực tiếp trên trang web
- Manual Mode và Auto Mode (với Gemini AI)
- Xuất kết quả dạng JSON/CSV
- Giao diện đơn giản, dễ sử dụng

Xem hướng dẫn chi tiết tại: [Chrome Extension README](chrome_extension/README.md)

---

## Cài đặt

### Yêu cầu hệ thống

- Python 3.8 trở lên
- CUDA-compatible GPU (khuyến nghị cho training)
- 4GB RAM trở lên

### Bước 1: Clone repository

```bash
git clone https://github.com/doananhhung/NER_Covid19.git
cd vietnamese_covid_ner
```

### Bước 2: Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Setup VnCoreNLP

Tải và giải nén models cho VnCoreNLP:

```bash
python setup_vncorenlp.py
```

Script này sẽ tự động tải VnCoreNLP models vào thư mục `vncorenlp_models/`.

### Bước 5: Cấu hình Gemini API (Optional - cho Auto Mode)

Nếu muốn sử dụng chế độ Auto Mode:

1. Lấy API key từ [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Tạo file `.streamlit/secrets.toml`:

```toml
[gemini]
api_key = "your-gemini-api-key-here"
```

**Lưu ý:** Không commit file `secrets.toml` lên repository.

---

## Sử dụng

### 1. Training - Huấn luyện mô hình

Huấn luyện mô hình NER từ đầu:

```bash
python src/train.py
```

**Cấu hình training** có thể thay đổi trong `src/config.py`:
- `MAX_LEN`: Độ dài tối đa sequence (mặc định: 256)
- `TRAIN_BATCH_SIZE`: Batch size training (mặc định: 8)
- `EPOCHS`: Số epochs (mặc định: 5)
- `LEARNING_RATE`: Learning rate (mặc định: 3e-5)

Mô hình sau khi huấn luyện sẽ được lưu trong `models/phobert-ner-covid/`.

### 2. Evaluation - Đánh giá mô hình

Đánh giá mô hình trên test set:

```bash
python src/evaluate.py
```

Kết quả sẽ hiển thị:
- Overall metrics (Precision, Recall, F1-score)
- Per-entity metrics
- Confusion matrix (optional)

### 3. Inference - Dự đoán

Sử dụng mô hình để dự đoán entities từ văn bản:

```python
from src.inference import NERPredictor

# Khởi tạo predictor
predictor = NERPredictor(
    model_path="models/phobert-ner-covid",
    use_word_segmentation=True
)

# Dự đoán
text = "Bệnh nhân 123 là Nguyễn Văn A, 35 tuổi, nam, sống tại Hà Nội."
predictions = predictor.predict(text)

# Hiển thị kết quả
for pred in predictions:
    print(f"{pred['word']}: {pred['tag']}")
```

### 4. Chạy Web Application

#### Cách 1: Sử dụng script wrapper (khuyến nghị)

```bash
python run_app.py
```

Script này tự động:
- Phát hiện và sử dụng virtual environment nếu có
- Thiết lập đúng working directory
- Chạy Streamlit app với cấu hình tối ưu

#### Cách 2: Chạy trực tiếp với Streamlit

```bash
streamlit run app/app_combined.py
```

**Truy cập ứng dụng:** Mở trình duyệt tại `http://localhost:8501`

#### Sử dụng Web App

1. **Manual Mode**:
   - Nhập văn bản về 1 bệnh nhân
   - Xem entities được highlight
   - Xem thông tin bệnh nhân được trích xuất

2. **Auto Mode** (cần Gemini API key):
   - Nhập văn bản dài chứa nhiều bệnh nhân
   - Hệ thống tự động tách và xử lý từng bệnh nhân
   - Xuất kết quả dưới dạng JSON/CSV

### 5. Sử dụng Chrome Extension

#### Cài đặt Extension

1. **Cài đặt backend dependencies:**
```bash
pip install -r backend_api/requirements_api.txt
```

2. **Khởi động Backend API Server:**
```bash
python run_extension_server.py
```
Server sẽ chạy tại `http://localhost:8000`

3. **Load Extension vào Chrome:**
   - Mở Chrome và truy cập `chrome://extensions/`
   - Bật "Developer mode"
   - Click "Load unpacked"
   - Chọn thư mục `chrome_extension/`

#### Sử dụng Extension

1. Click icon Extension trên toolbar
2. Chọn nguồn dữ liệu: "Xử lý toàn bộ trang web" hoặc "Nhập văn bản thủ công"
3. Chọn chế độ xử lý: Manual Mode hoặc Auto Mode
4. Click "Phân tích"
5. Xem kết quả và export CSV/JSON hoặc highlight trên trang

Chi tiết xem tại: [Chrome Extension README](chrome_extension/README.md)

---

## Cấu trúc dự án

```
vietnamese_covid_ner/
│
├── README.md                          # File này
├── requirements.txt                   # Python dependencies
├── run_app.py                        # Script wrapper để chạy Streamlit app
├── run_extension_server.py           # Script khởi động Backend API cho Extension
├── setup_vncorenlp.py                # Script setup VnCoreNLP
│
├── data/                             # Thư mục dữ liệu
│   └── raw/
│       └── PhoNER_COVID19/           # Dataset PhoNER_COVID19
│           ├── train_word.json       # Training set
│           ├── dev_word.json         # Development set
│           └── test_word.json        # Test set
│
├── models/                           # Thư mục lưu mô hình
│   └── phobert-ner-covid/            # Mô hình PhoBERT đã fine-tune
│       ├── config.json
│       ├── model.safetensors
│       ├── vocab.txt
│       └── ...
│
├── notebooks/                        # Jupyter notebooks
│   ├── Data_Exploration.ipynb        # Phân tích và khảo sát dữ liệu
│   └── Train_on_Colab_basic.ipynb    # Training trên Google Colab
│
├── src/                              # Source code chính
│   ├── __init__.py
│   ├── config.py                     # Cấu hình tập trung (paths, hyperparameters)
│   ├── dataset.py                    # PyTorch Dataset cho NER
│   ├── train.py                      # Script training
│   ├── evaluate.py                   # Script evaluation
│   ├── inference.py                  # NERPredictor class
│   ├── text_processor.py             # Xử lý văn bản tiếng Việt
│   │
│   └── patient_extraction/           # Module trích xuất thông tin bệnh nhân
│       ├── __init__.py
│       ├── entity_structures.py      # Định nghĩa data structures
│       ├── manual_extractor.py       # Trích xuất thủ công
│       └── gemini_splitter.py        # Tách văn bản với Gemini AI
│
├── app/                              # Web application
│   ├── __init__.py
│   ├── app_combined.py               # Streamlit app (Manual + Auto mode)
│   └── utils.py                      # Utility functions cho UI
│
├── backend_api/                      # Backend API cho Chrome Extension
│   ├── __init__.py
│   ├── main.py                       # FastAPI application
│   ├── api_models.py                 # Pydantic models cho API
│   └── requirements_api.txt          # Dependencies cho API server
│
├── chrome_extension/                 # Chrome Extension
│   ├── manifest.json                 # Extension configuration
│   ├── README.md                     # Hướng dẫn sử dụng Extension
│   ├── icons/                        # Extension icons
│   ├── popup/                        # Popup UI (HTML/CSS/JS)
│   ├── content/                      # Content scripts
│   ├── background/                   # Background service worker
│   └── shared/                       # Shared utilities
│
├── vncorenlp_models/                 # VnCoreNLP models
│   └── models/
│       └── wordsegmenter/            # Word segmentation models
│
└── .streamlit/                       # Cấu hình Streamlit
    └── secrets.toml                  # API keys (không commit)
```

### Giải thích các file quan trọng

#### **src/config.py**
File cấu hình tập trung chứa:
- Đường dẫn files và thư mục
- Hyperparameters training
- Danh sách entities và tag mapping
- Cấu hình VnCoreNLP

#### **src/dataset.py**
Định nghĩa `NERDataset` (PyTorch Dataset):
- Load và tokenize dữ liệu
- Xử lý label alignment cho sub-word tokens
- Padding và truncation

#### **src/train.py**
Script training chính:
- Load dataset và mô hình PhoBERT
- Training loop với validation
- Lưu checkpoint tốt nhất

#### **src/evaluate.py**
Đánh giá mô hình:
- Tính toán metrics (seqeval)
- Per-entity performance
- Confusion matrix

#### **src/inference.py**
Class `NERPredictor` cho inference:
- Load mô hình đã train
- Tích hợp VnCoreNLP
- Xử lý sub-word tokens
- Trả về predictions

#### **src/patient_extraction/**
Module trích xuất thông tin bệnh nhân có cấu trúc:
- `entity_structures.py`: Định nghĩa `PatientRecord` dataclass
- `manual_extractor.py`: Logic trích xuất từ entities
- `gemini_splitter.py`: Tách văn bản nhiều bệnh nhân bằng Gemini AI

#### **app/app_combined.py**
Streamlit web application:
- Giao diện 2 tab (Manual/Auto Mode)
- Visualize entities với màu sắc
- Hiển thị thông tin bệnh nhân
- Xuất kết quả JSON/CSV

---

## Dataset

### PhoNER_COVID19

Dataset được sử dụng: **PhoNER_COVID19** - Một corpus tiếng Việt được gán nhãn thủ công cho bài toán NER trong domain COVID-19.

**Thống kê:**
- **Training set**: ~5,000 câu
- **Development set**: ~500 câu
- **Test set**: ~500 câu

**Nguồn:** [VinAI Research](https://github.com/VinAIResearch/PhoNER_COVID19)

**Format:** JSON với cấu trúc:
```json
{
  "id": "001",
  "words": ["Bệnh", "nhân", "123", "là", "Nguyễn", "Văn", "A"],
  "tags": ["O", "O", "B-PATIENT_ID", "O", "B-NAME", "I-NAME", "I-NAME"]
}
```

---

## Mô hình

### Architecture

```
Input Text (Vietnamese)
    ↓
VnCoreNLP Word Segmentation
    ↓
PhoBERT Tokenizer (BPE)
    ↓
PhoBERT Base Model (vinai/phobert-base)
    ↓
Linear Classification Head
    ↓
Predictions (BIO Tags)
```

### PhoBERT

- **Base Model**: `vinai/phobert-base`
- **Architecture**: RoBERTa-based, pre-trained cho tiếng Việt
- **Vocab Size**: 64,000 BPE tokens
- **Hidden Size**: 768
- **Layers**: 12 transformer layers
- **Parameters**: ~135M

### Fine-tuning Strategy

1. **Freeze**: Không freeze bất kỳ layer nào (full fine-tuning)
2. **Learning Rate**: 3e-5 với linear warmup
3. **Batch Size**: 8 (train) / 4 (validation)
4. **Max Length**: 256 tokens
5. **Epochs**: 5 epochs
6. **Optimizer**: AdamW
7. **Label Smoothing**: Sử dụng -100 cho sub-word tokens

### Performance

Metrics trên test set (sau 5 epochs):

| Metric | Score |
|--------|-------|
| Overall Precision | ~88-92% |
| Overall Recall | ~86-90% |
| Overall F1 | ~87-91% |

**Lưu ý:** Kết quả cụ thể phụ thuộc vào hyperparameters và random seed.

---

## Công nghệ sử dụng

- **PyTorch** - Deep learning framework
- **Transformers** (Hugging Face) - Pre-trained models
- **VnCoreNLP** - Vietnamese NLP toolkit
- **seqeval** - Sequence labeling evaluation
- **Streamlit** - Web application framework
- **Google Generative AI** - Gemini API integration
- **pandas** - Data manipulation

---

## Hướng dẫn phát triển

### Thay đổi hyperparameters

Chỉnh sửa trong `src/config.py`:

```python
MAX_LEN = 256              # Tăng nếu văn bản dài hơn
TRAIN_BATCH_SIZE = 8       # Giảm nếu GPU out of memory
LEARNING_RATE = 3e-5       # Điều chỉnh để tối ưu training
EPOCHS = 5                 # Tăng để train lâu hơn
```

### Thêm entity mới

1. Cập nhật `UNIQUE_TAGS` trong `src/config.py`
2. Chuẩn bị dữ liệu với nhãn mới
3. Re-train mô hình

### Tích hợp vào hệ thống khác

Sử dụng `NERPredictor` class:

```python
from src.inference import NERPredictor

predictor = NERPredictor(
    model_path="models/phobert-ner-covid",
    use_word_segmentation=True
)

# API-style usage
def extract_entities(text: str):
    predictions = predictor.predict(text)
    # Process predictions
    return predictions
```

---

## Troubleshooting

### Lỗi khi chạy VnCoreNLP

**Vấn đề:** `FileNotFoundError: vncorenlp_models not found`

**Giải pháp:**
```bash
python setup_vncorenlp.py
```

### GPU Out of Memory

**Giải pháp:**
- Giảm `TRAIN_BATCH_SIZE` trong `src/config.py`
- Giảm `MAX_LEN`
- Sử dụng gradient accumulation

### Streamlit không chạy

**Vấn đề:** `ModuleNotFoundError: No module named 'streamlit'`

**Giải pháp:**
```bash
pip install streamlit
# hoặc
pip install -r requirements.txt
```

### Gemini API không hoạt động

**Kiểm tra:**
1. API key có đúng không?
2. File `.streamlit/secrets.toml` có tồn tại không?
3. Có kết nối internet không?

---

## Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

---

## License

Dự án này được phát hành dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

---

## Liên hệ

- **Repository**: [https://github.com/doananhhung/NER_Covid19](https://github.com/doananhhung/NER_Covid19)
- **Issues**: [https://github.com/doananhhung/NER_Covid19/issues](https://github.com/doananhhung/NER_Covid19/issues)

---

## Tài liệu tham khảo

1. **PhoBERT**: [https://github.com/VinAIResearch/PhoBERT](https://github.com/VinAIResearch/PhoBERT)
2. **PhoNER_COVID19**: [https://github.com/VinAIResearch/PhoNER_COVID19](https://github.com/VinAIResearch/PhoNER_COVID19)
3. **VnCoreNLP**: [https://github.com/vncorenlp/VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP)
4. **Transformers**: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)

---

## Changelog

### Version 1.0.0 (November 2025)
- Hoàn thiện hệ thống NER với PhoBERT
- Tích hợp VnCoreNLP cho word segmentation
- Xây dựng Web App với Manual và Auto Mode
- Tích hợp Gemini AI cho trích xuất tự động
- Module trích xuất thông tin bệnh nhân có cấu trúc
- Hỗ trợ xuất dữ liệu JSON/CSV

---

**Cảm ơn bạn đã sử dụng Vietnamese COVID-19 NER!** 🚀
