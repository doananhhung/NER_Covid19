# Vietnamese COVID-19 Named Entity Recognition (NER)

Hệ thống **Named Entity Recognition (NER)** cho văn bản tiếng Việt liên quan đến COVID-19, sử dụng mô hình **PhoBERT** để nhận diện và trích xuất thông tin bệnh nhân từ các văn bản y tế.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-orange.svg)](https://huggingface.co/transformers/)

## 📋 Mục lục

- [Giới thiệu](#giới-thiệu)
- [Entities được nhận diện](#entities-được-nhận-diện)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Dataset & Model](#dataset--model)
- [Technical Details](#technical-details)

---

## 🎯 Giới thiệu

Dự án này xây dựng một hệ thống NER hoàn chỉnh để tự động nhận diện và trích xuất thông tin từ các văn bản y tế tiếng Việt liên quan đến COVID-19.

### Công nghệ sử dụng

- **PhoBERT** (`vinai/phobert-base`) - Mô hình ngôn ngữ tiếng Việt pre-trained
- **VnCoreNLP** - Công cụ tách từ tiếng Việt
- **PhoNER_COVID19** - Dataset được gán nhãn cho bài toán NER
- **FastAPI** - Backend API server
- **Chrome Extension** - Giao diện người dùng chính
- **Streamlit** - Web app demo (optional)
- **Gemini AI** - Hỗ trợ tách văn bản nhiều bệnh nhân

### Ứng dụng thực tế

- ✅ Trích xuất thông tin bệnh nhân từ báo cáo y tế
- ✅ Tự động hóa việc ghi nhận thông tin trong hệ thống quản lý
- ✅ Hỗ trợ phân tích dữ liệu dịch bệnh COVID-19
- ✅ Highlight entities trực tiếp trên trang web
- ✅ Xuất dữ liệu sang Excel/CSV

---

## 🏷️ Entities được nhận diện

Hệ thống nhận diện **10 loại entities** theo định dạng BIO tagging:

| Entity Type | Mô tả | Ví dụ |
|-------------|-------|-------|
| **PATIENT_ID** | Mã số bệnh nhân | BN123, Bệnh nhân 456 |
| **NAME** | Họ và tên bệnh nhân | Nguyễn Văn A, Trần Thị B |
| **AGE** | Tuổi, độ tuổi | 35 tuổi, 40 |
| **GENDER** | Giới tính | nam, nữ |
| **JOB** | Nghề nghiệp | bác sĩ, công nhân, giáo viên |
| **LOCATION** | Địa điểm | Hà Nội, quận 1, phường Bến Nghé |
| **ORGANIZATION** | Tổ chức, cơ quan | Bệnh viện Bạch Mai, CDC |
| **SYMPTOM_AND_DISEASE** | Triệu chứng và bệnh | sốt, ho, COVID-19, khó thở |
| **TRANSPORTATION** | Phương tiện di chuyển | xe buýt, chuyến bay VN123, taxi |
| **DATE** | Ngày tháng, thời gian | 15/3/2021, ngày 20 tháng 4 |

**Định dạng BIO tagging:**
- `B-[ENTITY]`: Beginning - Token đầu tiên của entity
- `I-[ENTITY]`: Inside - Token tiếp theo của entity (cho multi-word entities)
- `O`: Outside - Không thuộc entity nào

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                      CHROME EXTENSION (UI)                      │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Floating    │  │  Side Panel  │  │  Highlight Entities   │  │
│  │   Button    │→ │   (450px)    │→ │   on Webpage         │  │
│  └─────────────┘  └──────────────┘  └───────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP API Calls
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND SERVER                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Endpoints:                                             │   │
│  │  • POST /api/ner/extract-manual  (1 bệnh nhân)         │   │
│  │  • POST /api/ner/extract-auto    (nhiều bệnh nhân)     │   │
│  │  • GET  /api/health              (health check)        │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ↓                    ↓                    ↓
┌───────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  NER Model    │  │ Gemini AI        │  │  VnCoreNLP       │
│  (PhoBERT)    │  │ (Text Splitter)  │  │  (Word Segment)  │
└───────────────┘  └──────────────────┘  └──────────────────┘
        ↓                    ↓                    ↓
┌─────────────────────────────────────────────────────────────────┐
│             PATIENT EXTRACTION & DEDUPLICATION                  │
│  • Smart Merge Algorithm (position-based)                      │
│  • Date Classification (9 types)                               │
│  • Entity Grouping & Validation                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Cài đặt

### Yêu cầu hệ thống

- Python 3.8 trở lên
- Chrome Browser (cho Extension)
- 4GB RAM (khuyến nghị 8GB)
- GPU (optional, cho training nhanh hơn)

### Bước 1: Clone repository

```bash
git clone https://github.com/doananhhung/NER_Covid19.git
cd vietnamese_covid_ner
```

### Bước 2: Tạo môi trường ảo

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Setup VnCoreNLP (tách từ tiếng Việt)

```bash
python setup_vncorenlp.py
```

Script này sẽ tự động:
- Download VnCoreNLP models
- Giải nén vào thư mục `vncorenlp_models/`
- Verify installation

### Bước 5: Chuẩn bị Model

**Option 1: Download pre-trained model (khuyên dùng)**

📥 **Download model đã train sẵn:**
- Link Google Drive: [https://drive.google.com/drive/folders/1GNf_xUUrswxe3feUWCaTyyLbzFnLfLHS?usp=drive_link](https://drive.google.com/drive/folders/1GNf_xUUrswxe3feUWCaTyyLbzFnLfLHS?usp=drive_link)
- Download toàn bộ thư mục `phobert-ner-covid/`
- Giải nén vào `models/phobert-ner-covid/`

**Cấu trúc thư mục sau khi giải nén:**
```
models/phobert-ner-covid/
├── config.json
├── model.safetensors
├── vocab.txt
├── bpe.codes
├── tokenizer_config.json
├── special_tokens_map.json
└── added_tokens.json
```

**Option 2: Train model từ đầu**
```bash
python src/train.py
```

Training config trong `src/config.py`:
- Epochs: 10
- Batch size: 16
- Learning rate: 2e-5
- Max length: 256 tokens

### Bước 6: (Optional) Cấu hình Gemini API

Cho chế độ Auto Mode (tách nhiều bệnh nhân):

```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "your-api-key-here"

# Windows CMD
set GEMINI_API_KEY=your-api-key-here

# Linux/Mac
export GEMINI_API_KEY="your-api-key-here"
```

Hoặc tạo file `.env`:
```env
GEMINI_API_KEY=your-api-key-here
```

Lấy API key tại: https://makersuite.google.com/app/apikey

---

## 🚀 Sử dụng

### 1. Chrome Extension (Khuyên dùng - Giao diện chính)

#### Khởi động Backend Server

```bash
python run_extension_server.py
```

Server chạy tại: `http://localhost:8000`

#### Cài đặt Extension

1. Mở Chrome, truy cập `chrome://extensions/`
2. Bật "Developer mode"
3. Click "Load unpacked"
4. Chọn thư mục `chrome_extension/`

#### Sử dụng

1. **Nút 🦠** xuất hiện ở góc phải mọi trang web
2. Click để mở **Side Panel**
3. Chọn nguồn:
   - "Trang web hiện tại" - Lấy text từ trang
   - "Nhập thủ công" - Paste văn bản
4. Chọn chế độ:
   - "Thủ công" - 1 bệnh nhân
   - "Tự động" - Nhiều bệnh nhân (cần Gemini)
5. Click "Phân tích"
6. Xem kết quả trong 2 tabs: Entities / Bệnh nhân
7. Export: Copy CSV / Download CSV / Highlight

**Xem chi tiết:** [Chrome Extension README](chrome_extension/README.md)

---

### 2. Streamlit Web App (Demo)

```bash
python run_app.py
```

App chạy tại: `http://localhost:8501`

**Tính năng:**
- Upload/paste văn bản
- 2 chế độ: Manual và Auto
- Hiển thị entities với màu highlight
- Export JSON/CSV

---

### 3. Python API (Programmatic)

#### Inference cơ bản

```python
from src.inference import NERPredictor

# Khởi tạo predictor
predictor = NERPredictor(
    model_path="models/phobert-ner-covid",
    use_word_segmentation=True
)

# Dự đoán
text = "Bệnh nhân BN123, nam, 35 tuổi, nghề nghiệp giáo viên."
entities = predictor.predict(text)

# entities: List[Dict]
# [
#   {"text": "BN123", "tag": "PATIENT_ID", "start": 11, "end": 16},
#   {"text": "nam", "tag": "GENDER", "start": 18, "end": 21},
#   ...
# ]
```

#### Trích xuất thông tin bệnh nhân

```python
from src.patient_extraction.manual_extractor import extract_single_patient

# Entities từ NER model
entities = predictor.predict(text)

# Trích xuất thông tin
patient = extract_single_patient(entities)

print(patient.patient_id)  # "BN123"
print(patient.name)         # "Nguyễn Văn A"
print(patient.age)          # "35"
print(patient.gender)       # "nam"
print(patient.locations)    # ["Hà Nội", "Quận 1"]
```

#### Auto Mode với Gemini

```python
from src.patient_extraction.gemini_splitter import split_text_with_gemini

# Văn bản nhiều bệnh nhân
long_text = """
Bệnh nhân BN123, nam, 35 tuổi...
Bệnh nhân BN124, nữ, 28 tuổi...
"""

# Tách văn bản
segments = split_text_with_gemini(long_text, api_key="your-key")

# Xử lý từng segment
patients = []
for segment in segments:
    entities = predictor.predict(segment)
    patient = extract_single_patient(entities)
    patients.append(patient)
```

---

## 📁 Cấu trúc dự án

```
vietnamese_covid_ner/
├── src/                              # Core NER modules
│   ├── config.py                     # Cấu hình (paths, hyperparameters)
│   ├── dataset.py                    # PyTorch Dataset cho training
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Evaluation với seqeval
│   ├── inference.py                  # NER Predictor
│   ├── text_processor.py             # VnCoreNLP wrapper
│   └── patient_extraction/           # Patient info extraction
│       ├── entity_structures.py      # Entity, PatientRecord dataclasses
│       ├── manual_extractor.py       # Logic trích xuất + smart merge
│       └── gemini_splitter.py        # Gemini text splitting
│
├── backend_api/                      # FastAPI server
│   ├── main.py                       # API endpoints
│   ├── api_models.py                 # Pydantic models
│   └── logger.py                     # Logging utilities
│
├── chrome_extension/                 # Chrome Extension (UI chính)
│   ├── manifest.json                 # Extension config
│   ├── content/                      # Content scripts
│   │   ├── floating-button.js        # Floating button UI
│   │   ├── side-panel.js            # Side panel logic
│   │   ├── side-panel.html          # Panel HTML
│   │   └── highlight.css            # Entity highlight styles
│   ├── background/                   # Service worker
│   └── shared/                       # Constants & utils
│
├── app/                              # Streamlit web app (demo)
│   ├── app_combined.py               # Main app
│   └── utils.py                      # UI utilities
│
├── data/                             # Dataset
│   └── raw/PhoNER_COVID19/          # Train/dev/test JSON
│
├── models/                           # Trained models
│   └── phobert-ner-covid/           # Model weights + tokenizer
│
├── logs/                             # API logs
│   └── ner_api.log                  # Full pipeline logs
│
├── tests/                            # Unit tests
│   └── test_merge_texts_smart.py    # Smart merge tests
│
├── docs/                             # Documentation
│   └── SMART_MERGE_IMPLEMENTATION.md # Technical docs
│
├── requirements.txt                  # Python dependencies
├── run_extension_server.py          # Launch FastAPI server
├── run_app.py                       # Launch Streamlit app
└── setup_vncorenlp.py               # VnCoreNLP setup script
```

---

## 📊 Dataset & Model

### Dataset: PhoNER_COVID19

- **Source**: [VinAI Research](https://github.com/VinAIResearch/PhoNER_COVID19)
- **Format**: JSON (word-level tokenization)
- **Size**: 
  - Train: ~5,000 sentences
  - Dev: ~700 sentences
  - Test: ~700 sentences
- **Entities**: 10 types (xem [section trên](#entities-được-nhận-diện))

### Model: PhoBERT-base

- **Base model**: `vinai/phobert-base`
- **Architecture**: RoBERTa-based, pre-trained on 20GB Vietnamese text
- **Fine-tuning**: Token classification head (11 classes: 10×2 BIO tags + O)
- **Performance** (trên test set):
  - Precision: ~87%
  - Recall: ~85%
  - F1-score: ~86%

### Training

```bash
python src/train.py
```

**Hyperparameters** (trong `src/config.py`):
```python
BATCH_SIZE = 16
MAX_LEN = 256
LEARNING_RATE = 2e-5
EPOCHS = 10
WARMUP_STEPS = 500
```

**GPU Memory**: ~6GB VRAM

---

## 🔧 Technical Details

### 1. Smart Merge Algorithm

**Vấn đề**: Duplicate entities (ví dụ: "BN123 BN123", "Nguyễn Văn A Nguyễn Văn A")

**Giải pháp**: Position-based deduplication
1. Sort entities by position
2. Group by proximity (gap < 5 chars = same mention)
3. Merge within group
4. Deduplicate mentions with `dict.fromkeys()`
5. Return first mention

**Code**: `src/patient_extraction/manual_extractor.py` → `_merge_texts_smart()`

**Tests**: `tests/test_merge_texts_smart.py` (12 test cases, all passed)

---

### 2. Date Classification

Hệ thống phân loại dates thành **9 loại**:

| Loại | Keywords | Ví dụ |
|------|----------|-------|
| admission | nhập viện, vào viện | ngày 15/3 nhập viện |
| discharge | xuất viện, ra viện | xuất viện ngày 20/3 |
| test | xét nghiệm, test | ngày 16/3 xét nghiệm |
| positive | dương tính, (+) | kết quả dương tính 17/3 |
| negative | âm tính, (-) | âm tính vào 25/3 |
| entry | nhập cảnh, vào VN | nhập cảnh ngày 10/3 |
| recovery | khỏi bệnh, hồi phục | khỏi bệnh 30/3 |
| death | tử vong, qua đời | tử vong ngày 1/4 |
| unknown | (không match) | ngày 15/3 |

**Code**: `src/patient_extraction/manual_extractor.py` → `_classify_date()`

---

### 3. Logging System

**File log**: `logs/ner_api.log`

**Ghi log:**
- Input text (preview 200 chars)
- NER results (max 20 entities)
- Gemini segments (full text)
- Patient records (all 16 fields)
- Processing time
- Errors with stack trace

**Code**: `backend_api/logger.py`

---

### 4. CSV Export với UTF-8 BOM

**Vấn đề**: Excel không hiển thị đúng tiếng Việt

**Giải pháp**: Thêm BOM (Byte Order Mark) `\uFEFF`

```javascript
const BOM = '\uFEFF';
const csv = BOM + header + '\n' + rows;
```

**Kết quả**: Excel mở đúng tiếng Việt có dấu

---

### 5. VnCoreNLP Integration

**Singleton pattern** để tránh load nhiều lần:

```python
class VietnameseTextProcessor:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
```

**Word segmentation**: Chuyển "công_việc giáo_viên" → ["công_việc", "giáo_viên"]

---

## 🧪 Testing

### Run unit tests

```bash
# Test smart merge algorithm
python -m pytest tests/test_merge_texts_smart.py -v

# Kết quả: 12/12 tests passed
```

### Manual testing

```bash
# Start server
python run_extension_server.py

# Test API
curl -X POST http://localhost:8000/api/ner/extract-manual \
  -H "Content-Type: application/json" \
  -d '{"text": "Bệnh nhân BN123, nam, 35 tuổi."}'
```

---

## 📝 Troubleshooting

### 1. Import Error: No module named 'src'

**Giải pháp**: Thêm project root vào PYTHONPATH
```bash
# Windows
set PYTHONPATH=%PYTHONPATH%;D:\path\to\vietnamese_covid_ner

# Linux/Mac
export PYTHONPATH="${PYTHONPATH}:/path/to/vietnamese_covid_ner"
```

### 2. VnCoreNLP không hoạt động

**Giải pháp**: 
```bash
python setup_vncorenlp.py
```

### 3. Model không load được

**Kiểm tra**: 
- Thư mục `models/phobert-ner-covid/` có đầy đủ files: `model.safetensors`, `config.json`, `vocab.txt`
- Nếu thiếu model, download từ: https://drive.google.com/drive/folders/1GNf_xUUrswxe3feUWCaTyyLbzFnLfLHS?usp=drive_link
- Hoặc train lại: `python src/train.py`

### 4. Gemini API lỗi

**Kiểm tra**:
- API key đúng format
- Đã set environment variable
- Quota chưa vượt giới hạn

### 5. Chrome Extension không kết nối server

**Kiểm tra**:
- Server đang chạy: http://localhost:8000/api/health
- CORS enabled trong FastAPI
- Port 8000 không bị chặn bởi firewall

---

## 📚 Documentation

- **Chrome Extension**: [chrome_extension/README.md](chrome_extension/README.md)
- **Quick Start**: [chrome_extension/QUICKSTART.md](chrome_extension/QUICKSTART.md)
- **Smart Merge**: [docs/SMART_MERGE_IMPLEMENTATION.md](docs/SMART_MERGE_IMPLEMENTATION.md)
- **Logging**: [logs/README.md](logs/README.md)
- **Streamlit Config**: [.streamlit/README.md](.streamlit/README.md)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👥 Authors

- **Đoàn Anh Hùng** - [GitHub](https://github.com/doananhhung)

---

## 🙏 Acknowledgments

- [VinAI Research](https://github.com/VinAIResearch) - PhoNER_COVID19 dataset & PhoBERT model
- [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) - Vietnamese NLP toolkit
- [Google Gemini](https://ai.google.dev/) - AI-powered text splitting

---

## 📞 Contact & Support

- **GitHub Issues**: [https://github.com/doananhhung/NER_Covid19/issues](https://github.com/doananhhung/NER_Covid19/issues)
- **Email**: doananhhung@example.com

---

**⭐ Nếu project hữu ích, hãy cho một star trên GitHub!**
