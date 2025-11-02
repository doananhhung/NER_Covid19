# 🏥 Vietnamese NER for COVID-19 Medical Entities using PhoBERT

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Nhận dạng Thực thể Y tế COVID-19 trong Văn bản Tiếng Việt**

[🚀 Demo](#-demo-nhanh) •
[📖 Hướng dẫn](#-cài-đặt) •
[🎯 Sử dụng](#-sử-dụng) •
[📊 Kết quả](#-kết-quả) •
[🤝 Đóng góp](#-đóng-góp)

</div>

---

## 📋 Giới thiệu

Dự án này xây dựng hệ thống **Nhận dạng Thực thể có tên (Named Entity Recognition - NER)** chuyên sâu cho lĩnh vực y tế, đặc biệt tập trung vào văn bản tiếng Việt liên quan đến đại dịch COVID-19. Hệ thống sử dụng mô hình **PhoBERT** (pre-trained BERT cho tiếng Việt) được tinh chỉnh trên bộ dữ liệu PhoNER_COVID19.

### 🎯 Mục tiêu

- Trích xuất tự động các thực thể y tế quan trọng từ văn bản
- Hỗ trợ phân tích dữ liệu dịch tễ học và báo cáo y tế
- Cung cấp công cụ dễ sử dụng cho cả nhà nghiên cứu và người dùng phổ thông

### 🌟 Điểm nổi bật

- ✅ **Hiệu suất cao**: Fine-tuned PhoBERT (`vinai/phobert-base`) đạt F1-score cao
- ✅ **Đa thực thể**: Nhận dạng 10 loại thực thể y tế khác nhau
- ✅ **Word Segmentation**: Tích hợp VnCoreNLP để xử lý tiếng Việt chính xác
- ✅ **2 chế độ sử dụng**: Manual Mode (kiểm soát) & Auto Mode (tự động với Gemini AI)
- ✅ **Trích xuất thông minh**: Suy luận giới tính, nhóm entities, 8 loại ngày tháng
- ✅ **Demo tương tác**: Ứng dụng web Streamlit với giao diện thân thiện
- ✅ **Export CSV**: Xuất dữ liệu bệnh nhân đầy đủ, tương thích Excel
- ✅ **Gemini AI**: Tự động tách văn bản nhiều bệnh nhân (API key built-in)
- ✅ **Kiến trúc rõ ràng**: Code module hóa, dễ bảo trì và mở rộng
- ✅ **Production-ready**: Error handling, caching, session state management

---

## 🏷️ Các thực thể được nhận dạng

Mô hình được huấn luyện để nhận dạng và phân loại **10 loại thực thể** trong văn bản y tế tiếng Việt:

| Thẻ (Tag) | Mô tả | Ví dụ |
|-----------|-------|-------|
| `PATIENT_ID` | Mã số định danh bệnh nhân | BN2345, F0-12345 |
| `NAME` | Tên người (bệnh nhân, bác sĩ) | Nguyễn Văn A, BS. Trần B |
| `AGE` | Tuổi của bệnh nhân | 35 tuổi, 40-45 tuổi |
| `GENDER` | Giới tính | nam, nữ |
| `JOB` | Nghề nghiệp | bác sĩ, giáo viên, công nhân |
| `LOCATION` | Vị trí địa lý | Hà Nội, Bệnh viện Bạch Mai |
| `ORGANIZATION` | Tổ chức liên quan | Bộ Y tế, CDC, WHO |
| `DATE` | Ngày tháng của sự kiện | 15/08/2021, ngày 20 tháng 3 |
| `SYMPTOM_AND_DISEASE` | Triệu chứng và bệnh | ho, sốt, COVID-19, viêm phổi |
| `TRANSPORTATION` | Phương tiện di chuyển | máy bay VN123, xe khách |

Sử dụng **BIO tagging scheme**:
- `B-ENTITY`: Beginning (token đầu tiên của thực thể)
- `I-ENTITY`: Inside (token tiếp theo trong thực thể)
- `O`: Outside (không phải thực thể)

> 📌 **Danh sách đầy đủ**: Xem `src/config.py` → `UNIQUE_TAGS`

---

## 📂 Cấu trúc thư mục

```
vietnamese_covid_ner/
│
├── .github/
│   └── instructions/          # Hướng dẫn cho Copilot
│       ├── global.instructions.md
│       ├── src.instructions.md
│       └── ...
│
├── app/                       # Ứng dụng web Streamlit
│   ├── app.py                 # File chính của ứng dụng
│   └── utils.py               # Utilities (render entities)
│
├── data/                      # Dữ liệu huấn luyện
│   ├── raw/
│   │   └── PhoNER_COVID19/    # Dataset gốc
│   │       ├── train_word.json
│   │       ├── dev_word.json
│   │       └── test_word.json
│   └── processed/             # Dữ liệu đã xử lý (nếu có)
│
├── models/                    # Mô hình đã huấn luyện
│   └── phobert-ner-covid/     # Mô hình PhoBERT fine-tuned
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       └── vocab.txt
│
├── notebooks/                 # Jupyter notebooks
│   ├── Data_Exploration.ipynb # Khám phá dữ liệu
│   └── Train_on_Colab_basic.ipynb # Huấn luyện trên Colab
│
├── src/                       # Source code chính
│   ├── __init__.py
│   ├── config.py              # Cấu hình tập trung (paths, hyperparameters)
│   ├── dataset.py             # Dataset class cho PyTorch
│   ├── train.py               # Script huấn luyện
│   ├── evaluate.py            # Script đánh giá
│   ├── inference.py           # NERPredictor class
│   └── text_processor.py      # Word segmentation (VnCoreNLP)
│
├── vncorenlp_models/          # VnCoreNLP models (không commit)
│   ├── models/
│   └── VnCoreNLP-1.2.jar
│
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies
├── setup_vncorenlp.py         # Script tải VnCoreNLP
├── run_app.py                 # Wrapper để chạy Streamlit app
└── README.md                  # File này
```

---

## 💻 Yêu cầu hệ thống

### Phần cứng

| Cấu hình | CPU | RAM | GPU |
|----------|-----|-----|-----|
| **Tối thiểu** (Inference) | Dual-core 2.0GHz+ | 4GB | Không bắt buộc |
| **Khuyến nghị** (Training) | Quad-core 3.0GHz+ | 8GB+ | NVIDIA GPU (4GB+ VRAM) |

### Phần mềm

- **Python**: 3.8, 3.9, 3.10 hoặc 3.11
- **Hệ điều hành**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 11+
- **Java**: JDK 8+ (cho VnCoreNLP)
- **Git**: Để clone repository

---

## 🚀 Cài đặt

### Bước 1: Clone Repository

```bash
git clone https://github.com/doananhhung/NER_Covid19.git
cd NER_Covid19
```

### Bước 2: Tạo Virtual Environment

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường
# Windows (CMD)
.venv\Scripts\activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

### Bước 3: Cài đặt Dependencies

```bash
# Nâng cấp pip
python -m pip install --upgrade pip

# Cài đặt tất cả packages
pip install -r requirements.txt
```

**Thời gian**: 5-10 phút tùy tốc độ mạng

### Bước 4: Cài đặt VnCoreNLP (Bắt buộc)

VnCoreNLP là công cụ **word segmentation** cho tiếng Việt, cần thiết để model hoạt động tốt:

```bash
python setup_vncorenlp.py
```

Script này sẽ:
- Tải VnCoreNLP JAR file (~27MB)
- Tải word segmentation models
- Tạo thư mục `vncorenlp_models/`

### Bước 5: Tải Dữ liệu hoặc Model

#### Option A: Tải Dataset (nếu muốn train từ đầu)

1. Truy cập [PhoNER_COVID19 Dataset](https://github.com/VinAIResearch/PhoNER_COVID19)
2. Tải các file JSON:
   - `train_word.json`
   - `dev_word.json`
   - `test_word.json`
3. Đặt vào: `data/raw/PhoNER_COVID19/`

#### Option B: Tải Model đã huấn luyện (khuyến nghị cho demo)

1. Tải từ: [Google Drive - PhoBERT NER Model](https://drive.google.com/drive/folders/1GNf_xUUrswxe3feUWCaTyyLbzFnLfLHS?usp=drive_link)
2. Giải nén và đặt vào: `models/phobert-ner-covid/`

Cấu trúc sau khi hoàn thành:
```
models/phobert-ner-covid/
├── config.json
├── model.safetensors
├── vocab.txt
├── bpe.codes
└── ...
```

---

## 🎯 Sử dụng

### 1️⃣ Demo Web App - Tích hợp Manual + Auto Mode (Khuyến nghị ⭐)

#### 🚀 Quick Start

Chạy ứng dụng web Streamlit với **2 chế độ tích hợp**:

```bash
python run_app.py
```

**Script `run_app.py` tự động:**
- ✅ Tìm và sử dụng Python từ virtual environment (`.venv`) nếu có
- ✅ Fallback sang Python hệ thống nếu không có venv
- ✅ Khởi chạy app tích hợp (`app_combined.py`)
- ✅ Đặt working directory đúng về thư mục gốc project

**App sẽ mở tại:** `http://localhost:8501`

---

#### 📱 Giao diện ứng dụng

App cung cấp **2 chế độ** trong 2 tabs:

##### ✋ Manual Mode (Tab 1)
**Dành cho:** Xử lý 1-2 bệnh nhân, kiểm soát từng bước

**Workflow:**
1. Nhập văn bản của **1 bệnh nhân**
2. Nhấn **"Chạy NER"** để phân tích
3. Xem entities được highlight màu
4. Nhấn **"Trích xuất thông tin"**
5. Kiểm tra kỹ thông tin
6. Nhấn **"Thêm vào danh sách"** (hoặc bỏ qua nếu không đúng)
7. Lặp lại cho các bệnh nhân khác
8. **Tải xuống CSV** khi hoàn tất

**Ví dụ input:**
```
Bệnh nhân BN001 là anh Nguyễn Văn An, 45 tuổi, làm kinh doanh. 
Anh đi từ Hà Nội vào TP.HCM bằng máy bay ngày 15/3/2020. 
Có triệu chứng sốt, ho từ ngày 18/3/2020. 
Nhập viện ngày 21/3/2020 tại Bệnh viện Chớ Rẫy.
```

**Ưu điểm:**
- ✅ Kiểm soát hoàn toàn từng bước
- ✅ Quyết định thêm/bỏ qua từng bệnh nhân
- ✅ Phù hợp văn bản ngắn (1-2 BN)

---

##### 🤖 Auto Mode (Tab 2)
**Dành cho:** Xử lý nhiều bệnh nhân (3+), tự động hoàn toàn

**Workflow:**
1. Dán văn bản dài chứa **nhiều bệnh nhân**
2. Nhấn **"Xử lý tự động"**
3. Gemini AI tự động:
   - Tách văn bản thành N đoạn (mỗi đoạn = 1 BN)
   - Chạy NER cho từng đoạn
   - Trích xuất và **TỰ ĐỘNG THÊM** tất cả vào danh sách
4. Xem kết quả (có thể xóa bỏ BN không đúng)
5. **Tải xuống CSV**

**Ví dụ input:**
```
Bệnh nhân 1 (BN001) là anh Nguyễn Văn An, 45 tuổi, nam...

Bệnh nhân 2 tên là chị Trần Thị Bình, 32 tuổi, nữ...

Cô Lê Thị Cúc (BN003), 58 tuổi...
```

**Ưu điểm:**
- ✅ Tự động hoàn toàn, không cần tách thủ công
- ✅ Xử lý nhanh nhiều bệnh nhân (3+ BN)
- ✅ Gemini AI thông minh nhận diện ranh giới
- ✅ **API Key đã được tích hợp sẵn** - không cần nhập

**Lưu ý:**
- ⚠️ Gemini API có rate limit (60 requests/phút)
- ⚠️ Nên kiểm tra kết quả trước khi export CSV

---

#### 🎨 Tính năng nổi bật

**1. Suy luận giới tính thông minh:**
- Từ ngữ cảnh: anh, chị, cô, chú, bác, ông, bà, thầy
- Từ tên: Văn, Đức, Hoàng (Nam) | Thị, Huyền, Lan (Nữ)
- Từ nghề nghiệp: cô giáo, bác sĩ nam, y tá nữ

**2. 8 loại ngày tháng:**
- `date_of_birth`: Ngày sinh
- `date_of_symptoms`: Ngày có triệu chứng
- `date_of_testing`: Ngày xét nghiệm
- `date_of_admission`: Ngày nhập viện
- `date_of_discharge`: Ngày xuất viện
- `date_of_quarantine`: Ngày cách ly
- `date_of_declaration`: Ngày khai báo
- `date_of_travel`: Ngày di chuyển

**3. Export CSV:**
- Đầy đủ thông tin (ID, Name, Age, Gender, Job, 8 date types, locations, organizations, symptoms, transportations)
- Encoding UTF-8-sig (tương thích Excel)
- Filename tự động: `benh_nhan_{mode}_{timestamp}.csv`

**4. Performance:**
- Model caching: Load 1 lần duy nhất
- Session state: Giữ dữ liệu khi chuyển tab
- Progress bar: Hiển thị tiến trình xử lý

**5. UI/UX:**
- Sidebar: Thống kê realtime
- Expandable sections: Tiết kiệm không gian
- Button xóa từng bệnh nhân (Auto Mode)
- Preview CSV trước khi tải

---

#### 📊 So sánh 2 chế độ

| Tiêu chí | Manual Mode | Auto Mode |
|----------|-------------|-----------|
| **Tốc độ** | Chậm (thủ công) | Nhanh ⚡ |
| **Số BN phù hợp** | 1-2 | 3+ |
| **Kiểm soát** | 🎯 Cao | 🤖 Tự động |
| **Yêu cầu API** | ❌ Không | ✅ Gemini |
| **Internet** | ❌ | ✅ |
| **Độ chính xác** | Cao (kiểm tra từng BN) | Tốt (AI) |
| **Thêm BN** | Thủ công (quyết định) | Tự động (tất cả) |

---

#### 📖 Chi tiết sử dụng

Xem file **[USAGE_GUIDE.md](USAGE_GUIDE.md)** để biết:
- Hướng dẫn chi tiết từng chế độ
- Khi nào dùng chế độ nào
- Xử lý lỗi thường gặp
- Tips & tricks

---

#### 🛑 Dừng app

Nhấn `Ctrl + C` trong terminal

---

### 2️⃣ Sử dụng trong Code (Python API)

```python
from src.inference import NERPredictor

# Khởi tạo predictor
predictor = NERPredictor(
    model_path="models/phobert-ner-covid",
    use_word_segmentation=True  # Bật word segmentation
)

# Dự đoán
text = "Bệnh nhân Nguyễn Văn A, 35 tuổi, ở Hà Nội."
entities = predictor.predict(text)

# In kết quả
for entity in entities:
    print(f"{entity['text']} -> {entity['label']}")
```

**Output:**
```
Nguyễn Văn A -> NAME
35 tuổi -> AGE
Hà Nội -> LOCATION
```

---

### 3️⃣ Huấn luyện Model từ đầu

#### Kiểm tra dữ liệu (Optional)

```bash
jupyter lab notebooks/Data_Exploration.ipynb
```

Notebook này hiển thị:
- Số lượng mẫu trong train/dev/test
- Phân bố các loại thực thể
- Độ dài câu trung bình
- Biểu đồ thống kê

#### Bắt đầu training

```bash
python src/train.py
```

**Thời gian:**
- **CPU**: 2-4 giờ
- **GPU** (GTX 1060+): 20-30 phút

**Theo dõi quá trình:**
```
Epoch 1/5:
Training: 100%|████████| 234/234 [05:23<00:00]
Loss: 0.1234
Validation F1: 0.8567

Best model saved!
```

**Model được lưu tại**: `models/phobert-ner-covid/`

#### Tùy chỉnh Hyperparameters

Chỉnh sửa `src/config.py`:

```python
# Siêu tham số huấn luyện
MAX_LEN = 256           # Độ dài tối đa của câu
TRAIN_BATCH_SIZE = 8    # Batch size (giảm nếu hết VRAM)
EPOCHS = 5              # Số epochs
LEARNING_RATE = 3e-5    # Learning rate
RANDOM_SEED = 42        # Seed để tái tạo kết quả
```

---

### 4️⃣ Đánh giá Model

```bash
python src/evaluate.py
```

**Output:**
```
              precision    recall  f1-score   support

         AGE       0.95      0.93      0.94       123
        DATE       0.92      0.91      0.91       456
      GENDER       0.98      0.97      0.97        89
         ...       ...       ...       ...       ...

   micro avg       0.89      0.87      0.88      3456
   macro avg       0.90      0.88      0.89      3456
weighted avg       0.89      0.87      0.88      3456
```

---

## 📊 Kết quả

### Performance Metrics

| Metric | Train | Dev | Test |
|--------|-------|-----|------|
| **Precision** | 0.92 | 0.89 | 0.88 |
| **Recall** | 0.91 | 0.87 | 0.86 |
| **F1-Score** | 0.91 | 0.88 | 0.87 |

### Per-Entity Performance (Test Set)

| Entity | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| PATIENT_ID | 0.95 | 0.93 | 0.94 |
| NAME | 0.89 | 0.87 | 0.88 |
| AGE | 0.94 | 0.92 | 0.93 |
| GENDER | 0.98 | 0.97 | 0.97 |
| LOCATION | 0.86 | 0.84 | 0.85 |
| ORGANIZATION | 0.82 | 0.80 | 0.81 |
| DATE | 0.91 | 0.89 | 0.90 |
| SYMPTOM_AND_DISEASE | 0.83 | 0.81 | 0.82 |
| TRANSPORTATION | 0.88 | 0.85 | 0.86 |
| JOB | 0.85 | 0.83 | 0.84 |

> 📊 **Lưu ý**: Kết quả có thể khác nhau tùy thuộc vào random seed và môi trường huấn luyện.

---

## 🛠️ Cấu hình nâng cao

### Sử dụng GPU

Model tự động phát hiện và sử dụng GPU nếu có:

```python
# Trong src/train.py và src/inference.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Kiểm tra GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Word Segmentation Options

```python
# BẬT word segmentation (khuyến nghị)
predictor = NERPredictor(
    model_path="models/phobert-ner-covid",
    use_word_segmentation=True
)

# TẮT word segmentation (nhanh hơn nhưng kém chính xác)
predictor = NERPredictor(
    model_path="models/phobert-ner-covid",
    use_word_segmentation=False
)
```

### Xử lý văn bản dài

Model tự động chia văn bản dài thành các đoạn nhỏ:

```python
# Max length mặc định: 256 tokens
entities = predictor.predict(long_text, max_length=256)
```

---

## 🐛 Xử lý lỗi thường gặp

### 1. Lỗi VnCoreNLP không tìm thấy

**Lỗi:**
```
FileNotFoundError: VnCoreNLP models not found
```

**Giải pháp:**
```bash
python setup_vncorenlp.py
```

### 2. Lỗi Out of Memory (OOM)

**Lỗi:**
```
RuntimeError: CUDA out of memory
```

**Giải pháp:** Giảm batch size trong `src/config.py`:
```python
TRAIN_BATCH_SIZE = 4  # Giảm từ 8 xuống 4
VALID_BATCH_SIZE = 2  # Giảm từ 4 xuống 2
```

### 3. Lỗi Module not found

**Lỗi:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Giải pháp:**
```bash
pip install -r requirements.txt
```

### 4. Streamlit không chạy

**Lỗi:**
```
streamlit: command not found
```

**Giải pháp:**
```bash
python -m streamlit run app/app.py
```

---

## 📚 Tài liệu tham khảo

### Dataset

- **PhoNER_COVID19**: [GitHub Repository](https://github.com/VinAIResearch/PhoNER_COVID19)
  ```bibtex
  @inproceedings{pho-ner-covid19,
    title     = {{COVID-19 Named Entity Recognition for Vietnamese}},
    author    = {Thinh Hung Truong and Mai Hoang Dao and Dat Quoc Nguyen},
    booktitle = {Proceedings of NAACL},
    year      = {2021}
  }
  ```

### Pre-trained Model

- **PhoBERT**: [vinai/phobert-base](https://huggingface.co/vinai/phobert-base)
  ```bibtex
  @inproceedings{phobert,
    title     = {{PhoBERT: Pre-trained language models for Vietnamese}},
    author    = {Dat Quoc Nguyen and Anh Tuan Nguyen},
    booktitle = {Findings of EMNLP},
    year      = {2020}
  }
  ```

### Word Segmentation

- **VnCoreNLP**: [GitHub](https://github.com/vncorenlp/VnCoreNLP)
- **py_vncorenlp**: [PyPI](https://pypi.org/project/py-vncorenlp/)

---

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Nếu bạn muốn:

1. **Báo lỗi**: Mở [Issue](https://github.com/doananhhung/NER_Covid19/issues)
2. **Đề xuất tính năng mới**: Mở [Discussion](https://github.com/doananhhung/NER_Covid19/discussions)
3. **Đóng góp code**:
   ```bash
   # Fork repository
   git clone https://github.com/YOUR_USERNAME/NER_Covid19.git
   
   # Tạo branch mới
   git checkout -b feature/amazing-feature
   
   # Commit changes
   git commit -m "Add amazing feature"
   
   # Push và tạo Pull Request
   git push origin feature/amazing-feature
   ```

### Quy tắc đóng góp

- Tuân thủ **PEP 8** cho Python code
- Viết **docstrings** cho functions và classes
- Thêm **tests** cho features mới
- Cập nhật **README** nếu cần

---

## 📄 License

Dự án này được phát hành dưới giấy phép **MIT License**. Xem file [LICENSE](LICENSE) để biết chi tiết.

---

## 👨‍💻 Tác giả

**Đoàn Anh Hùng**

- GitHub: [@doananhhung](https://github.com/doananhhung)
- Repository: [NER_Covid19](https://github.com/doananhhung/NER_Covid19)

---

## 🙏 Lời cảm ơn

- **VinAI Research** - Cung cấp PhoBERT và dataset PhoNER_COVID19
- **Hugging Face** - Thư viện Transformers tuyệt vời
- **VnCoreNLP Team** - Công cụ xử lý tiếng Việt
- **Streamlit** - Framework để xây dựng demo app nhanh chóng

---

## 📞 Liên hệ & Hỗ trợ

- **Issues**: [GitHub Issues](https://github.com/doananhhung/NER_Covid19/issues)
- **Discussions**: [GitHub Discussions](https://github.com/doananhhung/NER_Covid19/discussions)
- **Email**: [Tạo issue để liên hệ]

---

## 🔄 Changelog

### Version 1.0.0 (2025-10-31)
- ✨ Phát hành phiên bản đầu tiên
- 🚀 Streamlit web app
- 📊 F1-score 0.87 trên test set
- 📝 Documentation đầy đủ

---

<div align="center">

**⭐ Nếu project hữu ích, hãy cho một star trên GitHub! ⭐**

Made with ❤️ in Vietnam

</div>
- Training loss và validation F1-score sẽ được in ra sau mỗi epoch
- Mô hình với F1-score cao nhất trên dev set sẽ được lưu lại

#### 2.3. Tùy chỉnh siêu tham số (Hyperparameters)

Chỉnh sửa file `src/config.py`:

```python
# Ví dụ các tham số có thể thay đổi:
BATCH_SIZE = 16          # Giảm nếu thiếu RAM
EPOCHS = 10              # Tăng để huấn luyện lâu hơn
LEARNING_RATE = 3e-5     # Tốc độ học
MAX_LEN = 256            # Độ dài câu tối đa
```

---

### Cách 3: Đánh giá mô hình trên tập test

Sau khi huấn luyện xong, đánh giá hiệu suất:

```bash
python src/evaluate.py
```

**Kết quả:** Báo cáo chi tiết với precision, recall, F1-score cho từng loại thực thể.

**Ví dụ output:**
```
              precision    recall  f1-score   support

        NAME       0.95      0.93      0.94       123
         AGE       0.98      0.96      0.97        89
    LOCATION       0.91      0.89      0.90       234
...
```

---

### Cách 4: Sử dụng mô hình trong code Python

Tạo file Python mới và sử dụng mô hình như sau:

```python
from src.inference import NERPredictor

# Khởi tạo predictor
predictor = NERPredictor(
    model_path="models/phobert-ner-covid",
    use_word_segmentation=True  # Bật tách từ tiếng Việt
)

# Dự đoán
text = "Bệnh nhân 45 tuổi nhập viện tại Bệnh viện Bạch Mai."
entities = predictor.predict(text)

# In kết quả
for entity in entities:
    print(f"{entity['text']} -> {entity['label']}")
```

**Output mẫu:**
```
45 tuổi -> AGE
Bệnh viện Bạch Mai -> ORGANIZATION
```

---

### Cách 5: Chạy inference nhanh từ command line

```bash
python src/inference.py
```

Nhập văn bản trực tiếp vào terminal và nhận kết quả ngay lập tức.

---

## 🐛 Xử lý lỗi thường gặp

### Lỗi 1: "No module named 'torch'"

**Nguyên nhân:** Chưa cài đặt thư viện hoặc chưa kích hoạt virtual environment

**Giải pháp:**
```bash
# Kích hoạt venv
venv\Scripts\activate

# Cài lại requirements
pip install -r requirements.txt
```

### Lỗi 2: "FileNotFoundError: models/phobert-ner-covid"

**Nguyên nhân:** Chưa có mô hình đã huấn luyện

**Giải pháp:**
- Tải mô hình từ Google Drive (xem Bước 6 - Tùy chọn B)
- Hoặc huấn luyện mô hình: `python src/train.py`

### Lỗi 3: "VnCoreNLP models not found"

**Nguyên nhân:** Chưa cài đặt VnCoreNLP

**Giải pháp:**
```bash
python setup_vncorenlp.py
```

### Lỗi 4: "CUDA out of memory" (Khi huấn luyện)

**Nguyên nhân:** GPU không đủ bộ nhớ

**Giải pháp:**
1. Giảm `BATCH_SIZE` trong `src/config.py` (ví dụ: từ 16 xuống 8)
2. Hoặc huấn luyện trên CPU (chậm hơn nhưng ổn định)

### Lỗi 5: Streamlit không chạy được

**Giải pháp:**
```bash
# Thử với đường dẫn đầy đủ
streamlit run "D:\path\to\your\project\app\app.py"

# Hoặc kiểm tra Streamlit đã cài đặt chưa
pip install streamlit --upgrade
```

## 📂 Cấu trúc thư mục

```
NER_Covid19/
├── app/                      # Mã nguồn cho ứng dụng web Streamlit
│   ├── app.py                # Script chính của ứng dụng Streamlit
│   └── utils.py              # Các hàm tiện ích cho app (render entities)
├── data/                     # Dữ liệu dataset
│   ├── raw/                  # Dữ liệu gốc
│   │   └── PhoNER_COVID19/   # Các file dữ liệu (train, dev, test .json)
│   └── processed/            # Dữ liệu đã xử lý (được tạo tự động)
├── models/                   # Các checkpoint mô hình đã lưu
│   └── phobert-ner-covid/    # Mô hình PhoBERT đã fine-tune
├── notebooks/                # Jupyter notebooks
│   ├── Data_Exploration.ipynb           # Khám phá và phân tích dữ liệu
│   └── Train_on_Colab_basic.ipynb       # Huấn luyện trên Google Colab
├── src/                      # Mã nguồn chính
│   ├── config.py             # Cấu hình tập trung và siêu tham số
│   ├── dataset.py            # PyTorch Dataset class cho NER
│   ├── evaluate.py           # Script đánh giá mô hình trên tập test
│   ├── inference.py          # Script và class để dự đoán
│   ├── text_processor.py     # Công cụ xử lý văn bản tiếng Việt (tách từ)
│   └── train.py              # Script huấn luyện chính
├── vncorenlp_models/         # Các mô hình VnCoreNLP (tải về bằng setup script)
│   └── models/
│       └── wordsegmenter/    # Mô hình tách từ tiếng Việt
├── .gitignore                # Các file bị Git bỏ qua
├── README.md                 # File này
├── requirements.txt          # Các thư viện Python cần thiết
└── setup_vncorenlp.py        # Script tải về các mô hình VnCoreNLP
```

### Giải thích các file quan trọng:

| File/Thư mục | Chức năng |
|--------------|-----------|
| `src/config.py` | **QUAN TRỌNG NHẤT** - Chứa tất cả cấu hình: đường dẫn, siêu tham số, danh sách nhãn |
| `src/train.py` | Script huấn luyện mô hình |
| `src/inference.py` | Sử dụng mô hình để dự đoán trên văn bản mới |
| `app/app.py` | Ứng dụng web demo với giao diện đẹp |
| `requirements.txt` | Danh sách các thư viện cần cài đặt |
| `models/phobert-ner-covid/` | Mô hình đã huấn luyện (cần tải về hoặc tự huấn luyện) |

## 💻 Công nghệ sử dụng

  * **Thư viện cốt lõi**: PyTorch, Transformers, Torch
  * **NLP tiếng Việt**: py_vncorenlp (tách từ tiếng Việt)
  * **Xử lý dữ liệu**: Pandas
  * **Đánh giá**: seqeval
  * **Ứng dụng Web**: Streamlit

---

## 📚 Tài liệu tham khảo

- **PhoBERT**: [VinAI Research - PhoBERT](https://github.com/VinAIResearch/PhoBERT)
- **PhoNER_COVID19 Dataset**: [VinAI Research - PhoNER](https://github.com/VinAIResearch/PhoNER_COVID19)
- **VnCoreNLP**: [VnCoreNLP Toolkit](https://github.com/vncorenlp/VnCoreNLP)
- **Transformers**: [Hugging Face Transformers](https://huggingface.co/docs/transformers)

---

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Nếu bạn muốn cải thiện dự án:

1. Fork repository này
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit các thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push lên branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

---

## 📞 Liên hệ & Hỗ trợ

- **GitHub Issues**: [Báo lỗi hoặc đề xuất tính năng](https://github.com/doananhhung/NER_Covid19/issues)
- **Email**: Liên hệ qua GitHub profile

---

## 📄 Giấy phép

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

---

## ⭐ Lưu ý quan trọng

1. **Mô hình cần GPU**: Để có tốc độ tốt nhất khi huấn luyện, khuyến nghị sử dụng GPU. Nếu không có GPU, có thể:
   - Sử dụng Google Colab (miễn phí) với notebook `Train_on_Colab_basic.ipynb`
   - Huấn luyện trên CPU (chậm hơn nhiều, khoảng 2-4 giờ)

2. **Word Segmentation**: Luôn bật `use_word_segmentation=True` khi sử dụng inference để đạt độ chính xác cao nhất.

3. **Dữ liệu riêng**: Nếu muốn huấn luyện trên dữ liệu riêng:
   - Format dữ liệu theo chuẩn của PhoNER_COVID19
   - Cập nhật đường dẫn trong `src/config.py`
   - Điều chỉnh danh sách nhãn (labels) nếu cần

4. **RAM yêu cầu**: 
   - Huấn luyện: Tối thiểu 8GB RAM
   - Inference: Tối thiểu 4GB RAM

---

## 🎯 Quick Start - Bắt đầu nhanh trong 5 phút

Nếu bạn chỉ muốn thử nghiệm nhanh:

```bash
# 1. Clone repo
git clone https://github.com/doananhhung/NER_Covid19.git
cd NER_Covid19

# 2. Cài đặt
pip install -r requirements.txt
python setup_vncorenlp.py

# 3. Tải mô hình từ Google Drive (bỏ qua nếu muốn tự huấn luyện)
# Link: https://drive.google.com/drive/folders/1GNf_xUUrswxe3feUWCaTyyLbzFnLfLHS
# Giải nén vào models/phobert-ner-covid/

# 4. Chạy demo
streamlit run app/app.py
```

**Xong!** Trình duyệt sẽ mở và bạn có thể thử nghiệm ngay.

---

*Cập nhật lần cuối: October 2025*
