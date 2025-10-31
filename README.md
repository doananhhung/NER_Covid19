# Vietnamese NER for COVID-19 Medical Entities using PhoBERT

> **Repository**: [https://github.com/doananhhung/NER\_Covid19](https://github.com/doananhhung/NER_Covid19)

Dự án này tập trung vào Nhận dạng Thực thể có tên (Named Entity Recognition - NER) để trích xuất các thực thể y tế và dịch tễ học cụ thể từ văn bản tiếng Việt liên quan đến đại dịch COVID-19. Mô hình được xây dựng bằng cách tinh chỉnh **PhoBERT**, một mô hình BERT đơn ngữ tiên tiến cho tiếng Việt.

Dự án bao gồm các script cho khám phá dữ liệu, huấn luyện, đánh giá, và một ứng dụng web demo đơn giản sử dụng Streamlit.

## ✨ Tính năng nổi bật

  * **NER hiệu suất cao**: Tinh chỉnh PhoBERT (`vinai/phobert-base`) cho tác vụ NER tiếng Việt chuyên biệt.
  * **Nhận dạng đa thực thể**: Xác định 10 loại thực thể liên quan đến bối cảnh y tế và COVID-19.
  * **Cấu trúc rõ ràng**: Codebase được tổ chức tốt với sự phân tách rõ ràng giữa cấu hình, xử lý dữ liệu, huấn luyện và suy luận.
  * **Tái tạo được**: Bao gồm file requirements và random seed cố định để đảm bảo kết quả nhất quán.
  * **Demo tương tác**: Đi kèm với ứng dụng web Streamlit để dễ dàng kiểm tra và trực quan hóa kết quả.
  * **Hỗ trợ Colab**: Cung cấp Jupyter notebook để huấn luyện mô hình trên tài nguyên GPU miễn phí của Google Colab.

## 🏷️ Các thực thể được nhận dạng

Mô hình được huấn luyện để nhận dạng và phân loại các thực thể sau:

| Thẻ (Tag)                 | Mô tả                                     |
| ------------------------- | ----------------------------------------- |
| `PATIENT_ID`              | Mã số định danh bệnh nhân                 |
| `SYMPTOM_AND_DISEASE`     | Triệu chứng và bệnh được đề cập           |
| `LOCATION`                | Vị trí địa lý (thành phố, bệnh viện)      |
| `DATE`                    | Ngày tháng của sự kiện (ví dụ: ngày nhập viện) |
| `ORGANIZATION`            | Tổ chức liên quan (ví dụ: Bộ Y tế)       |
| `AGE`                     | Tuổi của bệnh nhân                        |
| `GENDER`                  | Giới tính của bệnh nhân                   |
| `NAME`                    | Tên của cá nhân                           |
| `TRANSPORTATION`          | Phương tiện di chuyển được sử dụng        |
| `JOB`                     | Nghề nghiệp của bệnh nhân                 |

*Danh sách này được định nghĩa trong `src/config.py`.*

## 🚀 Hướng dẫn sử dụng cho người mới bắt đầu

Hướng dẫn này dành cho người chưa có kinh nghiệm với Python hoặc Machine Learning. Hãy làm theo từng bước một cách cẩn thận.

### Bước 1: Cài đặt Python

**Yêu cầu hệ thống:**
  * Python 3.8 hoặc cao hơn
  * Ít nhất 4GB RAM
  * Khoảng 2GB dung lượng đĩa trống

**Cài đặt Python:**

1. **Windows:**
   - Tải Python từ [python.org](https://www.python.org/downloads/)
   - Chạy file cài đặt và **QUAN TRỌNG**: Tích vào ô "Add Python to PATH"
   - Nhấn "Install Now"

2. **Kiểm tra cài đặt:**
   ```bash
   python --version
   ```
   Bạn sẽ thấy output như: `Python 3.8.10` hoặc cao hơn

### Bước 2: Tải mã nguồn về máy

**Cách 1: Sử dụng Git (Khuyến nghị)**

```bash
# Cài đặt Git nếu chưa có: https://git-scm.com/downloads
git clone https://github.com/doananhhung/NER_Covid19.git
cd NER_Covid19
```

**Cách 2: Tải file ZIP**

1. Truy cập [https://github.com/doananhhung/NER_Covid19](https://github.com/doananhhung/NER_Covid19)
2. Nhấn nút "Code" > "Download ZIP"
3. Giải nén file ZIP vào thư mục bạn muốn

### Bước 3: Tạo môi trường ảo (Virtual Environment)

Môi trường ảo giúp cách ly các package Python của dự án này với hệ thống.

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows (CMD):
venv\Scripts\activate

# Trên Windows (PowerShell):
venv\Scripts\Activate.ps1

# Trên Linux/macOS:
source venv/bin/activate
```

**Lưu ý:** Sau khi kích hoạt, bạn sẽ thấy `(venv)` xuất hiện ở đầu dòng lệnh.

### Bước 4: Cài đặt các thư viện cần thiết

```bash
# Nâng cấp pip lên phiên bản mới nhất
python -m pip install --upgrade pip

# Cài đặt tất cả thư viện cần thiết
pip install -r requirements.txt
```

**Thời gian:** Quá trình này có thể mất 5-10 phút tùy thuộc vào tốc độ internet.

### Bước 5: Cài đặt VnCoreNLP (Bắt buộc)

VnCoreNLP là công cụ tách từ tiếng Việt, cần thiết cho mô hình hoạt động chính xác.

```bash
python setup_vncorenlp.py
```

**Kết quả:** Thư mục `vncorenlp_models/` sẽ được tạo ra với các file mô hình bên trong.

### Bước 6: Tải dữ liệu huấn luyện

**Tùy chọn A: Tải dataset PhoNER_COVID19**

1. Truy cập [PhoNER_COVID19 Dataset](https://github.com/VinAIResearch/PhoNER_COVID19)
2. Tải các file: `train_word.json`, `dev_word.json`, `test_word.json`
3. Đặt vào thư mục: `data/raw/PhoNER_COVID19/`

**Tùy chọn B: Sử dụng mô hình đã huấn luyện sẵn (Khuyến nghị cho người mới)**

Nếu bạn chỉ muốn dùng thử mà không muốn huấn luyện lại:

1. Tải mô hình từ: **[Google Drive](https://drive.google.com/drive/folders/1GNf_xUUrswxe3feUWCaTyyLbzFnLfLHS?usp=drive_link)**
2. Giải nén và đặt thư mục `phobert-ner-covid` vào `models/`

Cấu trúc sau khi hoàn thành:
```
models/
└── phobert-ner-covid/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    └── ... (các file khác)
```

## 🛠️ Hướng dẫn sử dụng chi tiết

### Cách 1: Chạy ứng dụng Web Demo (Dễ nhất - Khuyến nghị cho người mới)

Đây là cách nhanh nhất để trải nghiệm mô hình mà không cần hiểu về code.

```bash
streamlit run app/app.py
```

**Hoặc nếu gặp lỗi, thử:**

```bash
streamlit run "d:\đường\dẫn\đầy\đủ\đến\app\app.py"
```

**Sau khi chạy:**
1. Trình duyệt sẽ tự động mở (hoặc truy cập `http://localhost:8501`)
2. Nhập văn bản tiếng Việt về COVID-19 vào ô text
3. Nhấn nút "Phân tích"
4. Xem kết quả với các thực thể được đánh dấu màu

**Ví dụ văn bản để thử:**
```
Bệnh nhân nữ 35 tuổi, mã số BN2345, quê ở Hà Nội, nhập viện ngày 15/08/2021 với triệu chứng ho và sốt.
```

**Dừng ứng dụng:** Nhấn `Ctrl + C` trong terminal

---

### Cách 2: Huấn luyện mô hình từ đầu (Dành cho người muốn tùy chỉnh)

**Lưu ý:** Cần có dữ liệu huấn luyện (xem Bước 6 phía trên)

#### 2.1. Khám phá dữ liệu (Tùy chọn)

Mở notebook để xem thống kê dữ liệu:

```bash
jupyter lab notebooks/Data_Exploration.ipynb
```

#### 2.2. Bắt đầu huấn luyện

```bash
python src/train.py
```

**Thời gian:** 
- Với CPU: 2-4 giờ
- Với GPU: 20-30 phút

**Kết quả:** Mô hình tốt nhất sẽ được lưu tại `models/phobert-ner-covid/`

**Theo dõi quá trình:**
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
