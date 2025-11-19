# Architecture Diagrams

Thư mục này chứa các Mermaid diagrams mô tả kiến trúc hệ thống Vietnamese COVID-19 NER.

## 📊 Danh sách Diagrams

### 1. `architecture.mermaid` - Kiến trúc chi tiết
Diagram tổng quan chi tiết về toàn bộ hệ thống, bao gồm:
- Chrome Extension components (Floating Button, Side Panel, Content Script)
- FastAPI Backend endpoints và processing pipeline
- ML Model integration (PhoBERT, VnCoreNLP)
- External services (Gemini AI)
- Data flow giữa các components

**Sử dụng để:** Hiểu toàn bộ hệ thống và cách các components tương tác

---

### 2. `architecture-simple.mermaid` - Kiến trúc đơn giản
Diagram tổng quan ở mức cao (high-level) với 3 phần chính:
- Client Side (Chrome Extension + Website)
- Server Side (FastAPI + ML Pipeline)
- External Services (Gemini AI)

**Sử dụng để:** Trình bày nhanh, dễ hiểu cho người không kỹ thuật

---

### 3. `sequence-manual-mode.mermaid` - Flow Manual Mode
Sequence diagram mô tả luồng xử lý khi user sử dụng Manual Mode (1 bệnh nhân):
1. Mở Extension
2. Lấy text từ webpage
3. Phân tích với Backend API
4. Server processing (VnCoreNLP → PhoBERT → Extractor)
5. Hiển thị kết quả
6. Highlight entities trên page
7. Export CSV

**Sử dụng để:** Hiểu chi tiết từng bước xử lý, timing, message passing

---

### 4. `sequence-auto-mode.mermaid` - Flow Auto Mode
Sequence diagram mô tả luồng xử lý khi user sử dụng Auto Mode (nhiều bệnh nhân):
1. User input văn bản nhiều bệnh nhân
2. Backend gọi Gemini AI để tách text
3. Xử lý từng segment riêng biệt
4. Deduplication
5. Trả về list patients
6. User có thể xóa bệnh nhân sai
7. Export CSV

**Sử dụng để:** Hiểu cách xử lý multi-patient text, vai trò của Gemini AI

---

### 5. `ml-pipeline.mermaid` - ML Processing Pipeline
Diagram mô tả chi tiết pipeline xử lý ML từ raw text → structured data:
1. **Preprocessing:** VnCoreNLP word segmentation
2. **NER Model:** PhoBERT tokenization → inference → word alignment
3. **Patient Extraction:** Grouping → Smart Merge → Date Classification

**Sử dụng để:** Hiểu technical details của ML pipeline, các algorithms

---

### 6. `deployment-architecture.mermaid` - Deployment Architecture
Diagram mô tả deployment và setup:
- Development environment (Git)
- Local server setup (FastAPI, models, data)
- Client side setup (Chrome Extension)
- External dependencies (HuggingFace, Gemini)
- User types

**Sử dụng để:** Setup hệ thống, deployment planning, infrastructure

---

## 🔧 Cách xem Diagrams

### Option 1: Visual Studio Code (Recommended)
1. Install extension: **Markdown Preview Mermaid Support**
   ```
   ext install bierner.markdown-mermaid
   ```
2. Mở file `.mermaid`
3. Press `Ctrl+Shift+V` để preview

### Option 2: Mermaid Live Editor
1. Truy cập: https://mermaid.live/
2. Copy nội dung file `.mermaid`
3. Paste vào editor
4. Xem diagram real-time
5. Export as PNG/SVG/PDF

### Option 3: GitHub
- GitHub tự động render Mermaid diagrams trong Markdown
- View trực tiếp trên repository

### Option 4: IntelliJ IDEA / PyCharm
- Install plugin: **Mermaid**
- Open `.mermaid` file
- Preview pane tự động hiển thị

---

## 📝 Mermaid Syntax Reference

### Graph Types
```mermaid
graph TD    %% Top-Down
graph LR    %% Left-Right
graph TB    %% Top-Bottom (same as TD)
```

### Nodes
```mermaid
A[Square]
B(Rounded)
C([Stadium])
D{Diamond}
E>Flag]
```

### Arrows
```mermaid
A --> B     %% Solid arrow
A -.-> B    %% Dotted arrow
A ==> B     %% Thick arrow
```

### Styling
```mermaid
classDef className fill:#color,stroke:#color
class NodeA className
```

---

## 🎯 Sử dụng trong Trình bày

### Cho Nhà tuyển dụng:
1. **Bắt đầu:** `architecture-simple.mermaid` (overview)
2. **Deep dive:** `sequence-manual-mode.mermaid` (demo flow)
3. **Technical:** `ml-pipeline.mermaid` (ML expertise)

### Cho Documentation:
1. **README.md:** Embed `architecture-simple.mermaid`
2. **Technical docs:** Include `architecture.mermaid`
3. **User guide:** Use sequence diagrams

### Cho Presentation:
- Export as PNG (high resolution)
- Use in PowerPoint/Google Slides
- Add annotations/highlights

---

## 📦 Export Commands

### Export as PNG (high quality):
```bash
# Using mmdc (Mermaid CLI)
npm install -g @mermaid-js/mermaid-cli

mmdc -i architecture.mermaid -o architecture.png -w 2000 -H 1500
```

### Export as SVG (scalable):
```bash
mmdc -i architecture.mermaid -o architecture.svg
```

### Export as PDF:
```bash
mmdc -i architecture.mermaid -o architecture.pdf
```

---

## 🔄 Update Diagrams

Khi có thay đổi trong code/architecture:
1. Update tương ứng file `.mermaid`
2. Test render (Mermaid Live Editor)
3. Commit changes
4. Re-export images nếu cần

---

## 📚 Resources

- **Mermaid Documentation:** https://mermaid.js.org/
- **Mermaid Live Editor:** https://mermaid.live/
- **GitHub Mermaid Support:** https://github.blog/2022-02-14-include-diagrams-markdown-files-mermaid/
- **VS Code Extension:** https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid

---

## 💡 Tips

1. **Keep it simple:** Quá nhiều nodes → khó đọc
2. **Use subgraphs:** Group related components
3. **Color coding:** Different colors cho different layers
4. **Annotations:** Use notes để giải thích
5. **Consistent naming:** Đồng nhất tên components across diagrams

---

Tạo bởi: Đoàn Anh Hùng  
Project: Vietnamese COVID-19 NER  
Date: November 2025
