# HUONG DAN NHANH - CHROME EXTENSION

## Bat dau nhanh trong 3 buoc:

### 1. CAI DAT DEPENDENCIES

```bash
# Cai dat backend API dependencies
pip install fastapi uvicorn pydantic

# Hoac cai dat tu file
pip install -r backend_api/requirements_api.txt
```

### 2. KHOI DONG BACKEND SERVER

```bash
python run_extension_server.py
```

Doi cho den khi thay thong bao:
```
Server se chay tai: http://localhost:8000
```

**LUU Y:** Giu terminal nay mo! Server can chay trong khi su dung Extension.

### 3. CAI DAT EXTENSION

1. Mo Chrome browser
2. Go `chrome://extensions/` vao address bar
3. Bat "Developer mode" (o goc tren ben phai)
4. Click "Load unpacked"
5. Chon thu muc `chrome_extension/` trong project
6. Xong!

## Su dung Extension:

1. Click icon Extension tren toolbar Chrome
2. Neu status hien "Server san sang" (mau xanh) -> Co the su dung
3. Chon nguon du lieu: Trang web hoac Nhap thu cong
4. Chon che do: Manual (1 BN) hoac Auto (nhieu BN)
5. Click "Phan tich"
6. Xem ket qua va export neu can

## Troubleshooting:

### Loi "Server chua chay"
- Kiem tra terminal co dang chay `python run_extension_server.py` khong
- Kiem tra port 8000 co bi chiem boi ung dung khac khong

### Loi "Model khong the load"
- Kiem tra da train model chua: `python src/train.py`
- Hoac kiem tra thu muc `models/phobert-ner-covid/` co ton tai

### Extension khong hien thi
- Kiem tra da enable "Developer mode" trong chrome://extensions/
- Thu reload extension: Click icon reload trong chrome://extensions/

### Auto Mode khong hoat dong
- Can Gemini API key
- Set environment variable: `set GEMINI_API_KEY=your-key` (Windows CMD)
- Hoac `$env:GEMINI_API_KEY = "your-key"` (PowerShell)

## Lien he:

- Issues: https://github.com/doananhhung/NER_Covid19/issues
- README day du: chrome_extension/README.md
