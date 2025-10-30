# src/config.py
#
# File này chứa tất cả các cấu hình và siêu tham số (hyperparameters)
# cho dự án NER, giúp quản lý và thay đổi các thiết lập một cách tập trung.

import os

# --- 1. Đường dẫn (Paths) ---
# Đường dẫn tương đối từ thư mục gốc của dự án.
# Giả định các script trong `src/` được chạy từ thư mục gốc.
BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Thư mục gốc vietnamese-covid-ner/
DATA_DIR = os.path.join(BASE_PROJECT_DIR, 'data/raw/PhoNER_COVID19/')
TRAIN_FILE = os.path.join(DATA_DIR, 'train_word.json')
DEV_FILE = os.path.join(DATA_DIR, 'dev_word.json')
TEST_FILE = os.path.join(DATA_DIR, 'test_word.json')

# Thư mục để lưu các mô hình đã huấn luyện
MODEL_OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, 'models/phobert-ner-covid')


# --- 2. Cấu hình Mô hình (Model Configuration) ---
# Tên của mô hình pre-trained từ Hugging Face Hub
PRE_TRAINED_MODEL_NAME = "vinai/phobert-base"


# --- 3. Siêu tham số Huấn luyện (Training Hyperparameters) ---
# Độ dài tối đa của một chuỗi đầu vào.
# Dựa trên kết quả từ notebook EDA, 95%-99% câu có độ dài dưới 256.
# Chọn một giá trị là lũy thừa của 2 để tối ưu phần cứng (ví dụ: 128, 256, 512).
MAX_LEN = 256

# Kích thước batch size. Điều chỉnh tùy theo VRAM của GPU.
# Bắt đầu với giá trị nhỏ và tăng dần.
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4

# Số epochs để huấn luyện
EPOCHS = 5

# Learning rate cho optimizer
LEARNING_RATE = 3e-5

# Seed để đảm bảo kết quả có thể tái lập
RANDOM_SEED = 42


# --- 4. Cấu hình Nhãn (Tag Configuration) ---
# *** ĐÃ CẬP NHẬT DỰA TRÊN KẾT QUẢ EDA ***
UNIQUE_TAGS = [
    'O',
    'B-AGE',
    'I-AGE',
    'B-DATE',
    'I-DATE',
    'B-GENDER',
    'B-JOB',
    'I-JOB',
    'B-LOCATION',
    'I-LOCATION',
    'B-NAME',
    'I-NAME',
    'B-ORGANIZATION',
    'I-ORGANIZATION',
    'B-PATIENT_ID',
    'I-PATIENT_ID',
    'B-SYMPTOM_AND_DISEASE',
    'I-SYMPTOM_AND_DISEASE',
    'B-TRANSPORTATION',
    'I-TRANSPORTATION'
]


# Tạo mapping từ nhãn sang ID và ngược lại
TAGS_TO_IDS = {tag: i for i, tag in enumerate(UNIQUE_TAGS)}
IDS_TO_TAGS = {i: tag for i, tag in enumerate(UNIQUE_TAGS)}

# Nhãn đặc biệt cho các sub-word token.
# Chúng ta sẽ gán ID -100 cho các token này để Pytorch bỏ qua chúng khi tính loss.
SUBWORD_TAG_ID = -100

