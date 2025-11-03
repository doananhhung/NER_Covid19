# Chrome Extension - Vietnamese COVID-19 NER v2.1# Chrome Extension - Vietnamese COVID-19 NER



Chrome Extension để nhận diện thực thể và trích xuất thông tin bệnh nhân COVID-19 từ văn bản tiếng Việt.Chrome Extension de nhan dien thuc the va trich xuat thong tin benh nhan COVID-19 tu van ban tieng Viet.



## Mục lục## Tinh nang



- [Tính năng](#tính-năng)- **Xac dinh NER entities** tu van ban trang web hoac van ban thu cong

- [Cài đặt](#cài-đặt)- **Trich xuat thong tin benh nhan** voi 2 che do:

- [Sử dụng](#sử-dụng)  - Manual Mode: Trich xuat 1 benh nhan

- [Cấu hình Gemini API](#cấu-hình-gemini-api)  - Auto Mode: Tu dong tach va trich xuat nhieu benh nhan (dung Gemini AI)

- [Troubleshooting](#troubleshooting)- **Highlight entities** truc tiep tren trang web

- [Changelog](#changelog)- **Xuat ket qua** dang JSON hoac CSV



---## Cai dat



## Tính năng### 1. Cai dat Dependencies Backend



### Phiên bản hiện tại: v2.1```bash

# Tu thu muc goc cua project

#### 🎨 Giao diệnpip install -r requirements.txt

- **Floating Button** - Nút tròn nằm góc phải màn hình, luôn hiển thị trên mọi trang webpip install -r backend_api/requirements_api.txt

- **Side Panel** - Panel trượt từ bên phải (450px), không chiếm không gian trang web```

- **Giữ nội dung** - Khi đóng panel chỉ ẩn đi, mở lại vẫn thấy kết quả cũ

### 2. Khoi dong Backend Server

#### 🔍 Xử lý NER

- **Nhận diện entities** từ văn bản trang web hoặc văn bản thủ công```bash

- **2 chế độ xử lý**:python run_extension_server.py

  - **Manual Mode** (Thủ công): Xử lý văn bản 1 bệnh nhân```

  - **Auto Mode** (Tự động): Tách và xử lý nhiều bệnh nhân (dùng Gemini AI)

- **Tự động tách từ** với VnCoreNLP cho tiếng ViệtServer se chay tai `http://localhost:8000`



#### ✨ Tính năng mới v2.1### 3. Cai dat Extension vao Chrome

- **🖍️ Highlight Entities**: Đánh dấu màu entities trực tiếp trên trang web

- **💾 Tải CSV**: Download file CSV với timestamp tự động1. Mo Chrome browser

- **📋 Copy CSV**: Copy nội dung CSV vào clipboard2. Truy cap `chrome://extensions/`

- **Xuất kết quả**: Cả 2 định dạng Entities và Patients3. Bat "Developer mode" (o goc tren ben phai)

4. Click "Load unpacked"

#### 📊 Xuất dữ liệu5. Chon thu muc `chrome_extension/` trong project

- **Entities CSV**: Danh sách tất cả entities (Text, Tag, Start, End)6. Extension se xuat hien trong toolbar

- **Patients CSV**: Thông tin chi tiết bệnh nhân (19 cột: ID, tên, tuổi, giới tính, nghề nghiệp, địa điểm, tổ chức, triệu chứng, phương tiện, 9 loại ngày, cảnh báo)

- **UTF-8 BOM encoding**: Hiển thị đúng tiếng Việt trong Excel## Su dung



---### Workflow co ban:



## Cài đặt1. **Khoi dong backend server** truoc khi su dung Extension

2. **Click icon Extension** tren toolbar de mo popup

### Bước 1: Cài đặt Dependencies Backend3. **Chon nguon du lieu**:

   - "Xu ly toan bo trang web": Lay text tu trang web hien tai

```bash   - "Nhap van ban thu cong": Tu nhap/paste van ban

# Từ thư mục gốc của project4. **Chon che do xu ly**:

pip install -r requirements.txt   - Manual Mode: Cho van ban 1 benh nhan

```   - Auto Mode: Cho van ban nhieu benh nhan (can Gemini API key)

5. **Click "Phan tich"** de xu ly

### Bước 2: Khởi động Backend Server6. **Xem ket qua** trong 2 tabs:

   - Tab "Entities": Danh sach entities da nhan dien

```bash   - Tab "Benh nhan": Thong tin benh nhan da trich xuat

python run_extension_server.py7. **Xuat ket qua**:

```   - Copy JSON

   - Download CSV

Server sẽ chạy tại `http://localhost:8000`   - Highlight tren trang (neu chon "Xu ly toan bo trang web")



**LƯU Ý:** Giữ terminal này mở! Server cần chạy trong khi sử dụng Extension.## Cau hinh Gemini API (cho Auto Mode)



### Bước 3: Cài đặt Extension vào ChromeAuto Mode su dung Gemini AI de tach van ban nhieu benh nhan.



1. Mở Chrome browser### Cach 1: Environment Variable (Khuyên dùng)

2. Truy cập `chrome://extensions/`

3. Bật "Developer mode" (ở góc trên bên phải)```bash

4. Click "Load unpacked"# Windows (PowerShell)

5. Chọn thư mục `chrome_extension/` trong project$env:GEMINI_API_KEY = "your-api-key-here"

6. Extension xuất hiện trên toolbar

# Windows (CMD)

---set GEMINI_API_KEY=your-api-key-here



## Sử dụng# Linux/Mac

export GEMINI_API_KEY="your-api-key-here"

### Workflow cơ bản```



#### 1. Khởi độngSau do khoi dong server:

- **Khởi động backend server** trước khi sử dụng Extension```bash

- Truy cập **bất kỳ trang web nào**python run_extension_server.py

- **Nút tròn 🦠** sẽ xuất hiện ở góc phải màn hình```



#### 2. Mở Side Panel### Cach 2: Hard-code trong code (Tam thoi)

- **Click nút 🦠** để mở Side Panel

- Panel trượt vào từ bên phảiSua file `backend_api/main.py`, them:

```python

#### 3. Chọn nguồn văn bảngemini_api_key_env = "your-api-key-here"

- **"Trang web hiện tại"**: Lấy text từ trang đang xem```

  - *Lưu ý: Chỉ khả dụng chế độ "Tự động"*

- **"Nhập thủ công"**: Tự nhập/paste văn bản vào textarea### Lay Gemini API Key

  - *Có thể chọn cả 2 chế độ "Thủ công" và "Tự động"*

1. Truy cap https://makersuite.google.com/app/apikey

#### 4. Chọn chế độ xử lý2. Dang nhap bang Google Account

- **"Thủ công (1 bệnh nhân)"**: Xử lý văn bản chỉ có 1 bệnh nhân (không cần Gemini)3. Click "Create API Key"

- **"Tự động (nhiều bệnh nhân)"**: Tách và xử lý nhiều bệnh nhân (cần Gemini API key)4. Copy key va luu lai



#### 5. Phân tích## Cau truc Thu muc

- Click nút **"Phân tích"**

- Đợi xử lý (hiển thị loading indicator)```

chrome_extension/

#### 6. Xem kết quả├── manifest.json              # Extension configuration

- **Tab "Entities"**: Danh sách các entities đã nhận diện├── icons/                     # Extension icons

- **Tab "Bệnh nhân"**: Thông tin chi tiết từng bệnh nhân├── popup/                     # Popup UI

│   ├── popup.html

#### 7. Xuất kết quả│   ├── popup.css

│   └── popup.js

**Trong tab "Entities":**├── content/                   # Content Scripts

- **🖍️ Highlight**: Đánh dấu màu entities trên trang web (chỉ khi chọn "Trang web hiện tại")│   ├── content.js

- **📋 Copy CSV**: Copy danh sách entities dạng CSV│   └── highlight.css

- **💾 Tải CSV**: Download file `covid19_entities_YYYY-MM-DDTHH-MM-SS.csv`├── background/                # Background Service Worker

│   └── background.js

**Trong tab "Bệnh nhân":**└── shared/                    # Shared utilities

- **📋 Copy CSV**: Copy thông tin bệnh nhân dạng CSV (có thể xóa bệnh nhân trước khi copy)    ├── constants.js

- **💾 Tải CSV**: Download file `covid19_patients_YYYY-MM-DDTHH-MM-SS.csv`    └── utils.js

- **✕ Xóa bệnh nhân**: Click nút ✕ ở góc phải mỗi card để ẩn bệnh nhân```

- **↶ Hoàn tác**: Undo xóa bệnh nhân gần nhất

## Troubleshooting

#### 8. Đóng panel

- Click nút **✕** ở góc trên panel### Server chua chay

- Hoặc click lại **floating button**

- Nội dung vẫn được giữ, mở lại sẽ thấy kết quả cũ**Trieu chung**: Status indicator hien "Server chua chay", button "Phan tich" bi disable



---**Giai phap**:

```bash

## Cấu hình Gemini APIpython run_extension_server.py

```

Auto Mode sử dụng Gemini AI để tách văn bản nhiều bệnh nhân.

### Loi "Model khong the load"

### Cách 1: Environment Variable (Khuyên dùng)

**Nguyen nhan**: Model chua duoc train hoac khong ton tai

```bash

# Windows (PowerShell)**Giai phap**:

$env:GEMINI_API_KEY = "your-api-key-here"- Kiem tra thu muc `models/phobert-ner-covid/` co ton tai khong

- Train model neu chua co: `python src/train.py`

# Windows (CMD)- Hoac download model da train

set GEMINI_API_KEY=your-api-key-here

### Loi "VnCoreNLP khong kha dung"

# Linux/Mac

export GEMINI_API_KEY="your-api-key-here"**Nguyen nhan**: VnCoreNLP chua duoc setup

```

**Giai phap**:

Sau đó khởi động server:```bash

```bashpython setup_vncorenlp.py

python run_extension_server.py```

```

### Extension khong highlight duoc tren trang

### Cách 2: File .env (Persistent)

**Nguyen nhan**: 

Tạo file `.env` ở thư mục gốc project:- Trang web co cau truc DOM phuc tap

```env- Chrome security policies

GEMINI_API_KEY=your-api-key-here

```**Giai phap**:

- Thu voi cac trang web khac

### Lấy Gemini API Key- Kiem tra Console log trong DevTools



1. Truy cập https://makersuite.google.com/app/apikey### Gemini API loi

2. Đăng nhập bằng Google Account

3. Click "Create API Key"**Trieu chung**: Auto Mode tra ve loi

4. Copy key và lưu lại

**Giai phap**:

---- Kiem tra API key hop le

- Kiem tra da set environment variable

## Troubleshooting- Kiem tra quota cua Gemini API



### ❌ "Server chưa chạy"## API Endpoints



**Triệu chứng**: Status indicator màu đỏ, button "Phân tích" bị disableBackend API cung cap cac endpoints:



**Giải pháp**:- `GET /api/health` - Health check

```bash- `POST /api/ner/predict` - NER co ban

python run_extension_server.py- `POST /api/ner/extract-manual` - Manual mode

```- `POST /api/ner/extract-auto` - Auto mode



Đợi đến khi thấy: `Server sẽ chạy tại: http://localhost:8000`Xem API docs tai: http://localhost:8000/docs



---## Luu y



### ❌ Lỗi "Model không thể load"- Extension chi hoat dong khi backend server dang chay

- Gemini API can key hop le cho Auto Mode

**Nguyên nhân**: Model chưa được train hoặc không tồn tại- Highlight feature hoat dong tot nhat voi cac trang co noi dung tieng Viet

- File CSV su dung encoding UTF-8-BOM de mo duoc trong Excel

**Giải pháp**:

- Kiểm tra thư mục `models/phobert-ner-covid/` có tồn tại không## Contact

- Train model nếu chưa có: `python src/train.py`

- Hoặc download model đã train từ repository- Repository: https://github.com/doananhhung/NER_Covid19

- Issues: https://github.com/doananhhung/NER_Covid19/issues

---

### ❌ Lỗi "VnCoreNLP không khả dụng"

**Nguyên nhân**: VnCoreNLP chưa được setup

**Giải pháp**:
```bash
python setup_vncorenlp.py
```

---

### ❌ Panel không hiện

**Giải pháp**:
- Kiểm tra Console (F12) xem có lỗi không
- Thử reload lại trang web (F5)
- Reload Extension trong `chrome://extensions/`
- Kiểm tra extension đã được enable

---

### ❌ Không highlight được trên trang

**Nguyên nhân**: 
- Trang web có cấu trúc DOM phức tạp
- Chrome security policies
- Entities không match với text trên trang

**Giải pháp**:
- Thử với các trang web khác
- Kiểm tra Console log trong DevTools (F12)
- Đảm bảo đã chọn "Trang web hiện tại" làm nguồn

---

### ❌ Gemini API lỗi

**Triệu chứng**: Auto Mode trả về lỗi

**Giải pháp**:
- Kiểm tra API key hợp lệ
- Kiểm tra đã set environment variable đúng cách
- Kiểm tra quota của Gemini API (đã vượt giới hạn miễn phí chưa)
- Thử restart server sau khi set environment variable

---

### ❌ Excel không hiển thị đúng tiếng Việt

**Giải pháp**: File CSV đã sử dụng UTF-8-BOM encoding, nếu vẫn lỗi:
- Mở file bằng Notepad++
- Menu: Encoding → Convert to UTF-8-BOM
- Save lại file

---

## Cấu trúc thư mục

```
chrome_extension/
├── manifest.json              # Extension configuration
├── icons/                     # Extension icons (16x16, 48x48, 128x128)
├── content/                   # Content Scripts (inject vào trang web)
│   ├── content.js            # Main content script
│   ├── floating-button.js    # Floating button UI
│   ├── floating-button.css   # Button styles
│   ├── side-panel.js         # Side panel logic
│   ├── side-panel.html       # Panel UI structure
│   ├── side-panel.css        # Panel styles
│   └── highlight.css         # Highlight entity styles
├── background/               # Background Service Worker
│   └── background.js         # Extension lifecycle management
└── shared/                   # Shared utilities
    ├── constants.js          # Constants & API config
    └── utils.js              # Utility functions
```

---

## Changelog

### v2.1 (Current)
✨ **Tính năng mới:**
- 🖍️ Highlight entities trực tiếp trên trang web
- 💾 Tải xuống CSV với timestamp tự động
- ✕ Xóa bệnh nhân không chính xác (soft delete)
- ↶ Hoàn tác xóa bệnh nhân

🎨 **Cải tiến giao diện:**
- Result actions bar với 3 nút (Highlight / Copy / Download)
- Disabled state cho Manual mode khi chọn "Trang web hiện tại"
- Delete button (✕) ở góc phải mỗi patient card
- Undo button với counter hiển thị số lượng đã xóa

🐛 **Bug fixes:**
- Fixed: Manual mode không disable khi khởi tạo panel
- Fixed: Duplicate entities trong output (implemented smart merge)
- Fixed: Excel không hiển thị tiếng Việt (UTF-8 BOM)

### v2.0
✨ **Tính năng mới:**
- Floating Button + Side Panel thay thế Popup
- Giữ nội dung khi đóng panel
- Copy CSV thay vì JSON
- Không đóng tự động khi click ra ngoài

### v1.0
- Popup UI cơ bản
- Manual và Auto mode
- Export JSON

---

## API Endpoints

Backend API cung cấp các endpoints:

- `GET /api/health` - Health check
- `POST /api/ner/predict` - NER cơ bản (raw entities)
- `POST /api/ner/extract-manual` - Manual mode (1 bệnh nhân)
- `POST /api/ner/extract-auto` - Auto mode (nhiều bệnh nhân với Gemini)

Xem API docs tại: http://localhost:8000/docs

---

## Lưu ý

- Extension chỉ hoạt động khi backend server đang chạy
- Gemini API cần key hợp lệ cho Auto Mode
- Highlight feature hoạt động tốt nhất với các trang có nội dung tiếng Việt
- File CSV sử dụng encoding UTF-8-BOM để mở được trong Excel
- Soft delete: Bệnh nhân bị xóa chỉ ẩn khỏi UI và không xuất trong CSV, không xóa khỏi dữ liệu

---

## Contact

- Repository: https://github.com/doananhhung/NER_Covid19
- Issues: https://github.com/doananhhung/NER_Covid19/issues
- Quick Start: [QUICKSTART.md](QUICKSTART.md)
