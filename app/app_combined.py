# app/app_combined.py
"""
Streamlit App - Tích hợp Manual Mode + Auto Mode
Một ứng dụng hoàn chỉnh với 2 chế độ trích xuất thông tin bệnh nhân
"""

import sys
import os
import streamlit as st
import pandas as pd
from datetime import datetime
import io

# Setup paths
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_FILE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Imports
from src.inference import NERPredictor
from src import config as ner_config
from src.patient_extraction.manual_extractor import extract_single_patient
from src.patient_extraction.gemini_splitter import split_text_with_gemini
from src.patient_extraction.entity_structures import PatientRecord

# Import render_entities từ utils
try:
    from app.utils import render_entities
except ImportError:
    # Nếu import lỗi, định nghĩa hàm render_entities đơn giản
    def render_entities(sentence, entities):
        """Fallback render function"""
        st.write("**Entities found:**")
        for ent in entities:
            st.write(f"- {ent['text']} ({ent['tag']})")



# SHARED FUNCTIONS


@st.cache_resource
def load_ner_model():
    """Tải mô hình NER với cache - dùng chung cho cả 2 chế độ"""
    try:
        predictor = NERPredictor(
            model_path=ner_config.MODEL_OUTPUT_DIR,
            use_word_segmentation=True
        )
        if predictor.model is None:
            return None
        return predictor
    except Exception as e:
        st.error(f"❌ Lỗi khi tải mô hình: {e}")
        return None


def display_patient_record(record: PatientRecord, index: int = None):
    """
    Hiển thị thông tin của 1 PatientRecord
    
    Args:
        record: PatientRecord object
        index: Số thứ tự (optional)
    """
    # Header
    if index is not None:
        st.subheader(f"👤 Bệnh nhân #{index}")
    else:
        st.subheader(f"👤 {record.patient_id or record.name or 'Unknown'}")
    
    # Layout 2 cột
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 Thông tin cơ bản:**")
        st.write(f"- **Mã BN:** {record.patient_id or 'N/A'}")
        st.write(f"- **Họ tên:** {record.name or 'N/A'}")
        st.write(f"- **Tuổi:** {record.age or 'N/A'}")
        st.write(f"- **Giới tính:** {record.gender or 'N/A'}")
        st.write(f"- **Nghề nghiệp:** {record.job or 'N/A'}")
    
    with col2:
        st.markdown("**📅 Thông tin thời gian:**")
        has_dates = False
        for date_type, dates in record.dates.items():
            if dates:
                has_dates = True
                display_name = date_type.replace('_', ' ').title()
                st.write(f"- **{display_name}:** {', '.join(dates)}")
        if not has_dates:
            st.write("- Chưa có thông tin thời gian")
    
    # Thông tin bổ sung
    if record.locations:
        st.markdown("**📍 Địa điểm:**")
        st.write(", ".join(record.locations))
    
    if record.organizations:
        st.markdown("**🏥 Tổ chức:**")
        st.write(", ".join(record.organizations))
    
    if record.symptoms_and_diseases:
        st.markdown("**🩺 Triệu chứng/Bệnh:**")
        st.write(", ".join(record.symptoms_and_diseases))
    
    if record.transportations:
        st.markdown("**🚗 Phương tiện:**")
        st.write(", ".join(record.transportations))
    
    # Metadata
    st.markdown("---")
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.metric("Độ tin cậy", f"{record.confidence:.2%}")
    with col_meta2:
        if record.warnings:
            st.warning(f"⚠️ {len(record.warnings)} cảnh báo")
    
    # Warnings detail
    if record.warnings:
        with st.expander("⚠️ Chi tiết cảnh báo"):
            for warning in record.warnings:
                st.warning(warning)


def create_csv_from_records(records: list) -> bytes:
    """
    Tạo CSV từ list of PatientRecord
    
    Returns:
        bytes: CSV data
    """
    records_dict = [record.to_dict() for record in records]
    df = pd.DataFrame(records_dict)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    return csv_buffer.getvalue().encode('utf-8-sig')


# ============================================================================
# MANUAL MODE TAB
# ============================================================================

def manual_mode_tab(predictor: NERPredictor):
    """Tab cho chế độ thủ công"""
    
    st.markdown("""
    ### Chế độ Thủ công
    
    **Hướng dẫn:** Nhập văn bản 1 bệnh nhân → Chạy NER → Trích xuất → Thêm vào danh sách → Tải CSV
    """)
    
    # Initialize session state for manual mode
    if 'manual_ner_results' not in st.session_state:
        st.session_state.manual_ner_results = None
    if 'manual_current_record' not in st.session_state:
        st.session_state.manual_current_record = None
    if 'manual_patient_list' not in st.session_state:
        st.session_state.manual_patient_list = []
    if 'manual_raw_text' not in st.session_state:
        st.session_state.manual_raw_text = ""
    
    # Sidebar stats
    with st.sidebar:
        st.markdown("---")
        st.header("📊 Thống kê (Manual)")
        st.metric("Số bệnh nhân đã thêm", len(st.session_state.manual_patient_list))
        
        if st.button("🗑️ Xóa danh sách (Manual)", key="clear_manual"):
            st.session_state.manual_patient_list = []
            st.session_state.manual_ner_results = None
            st.session_state.manual_current_record = None
            st.session_state.manual_raw_text = ""
            st.rerun()
    
    st.markdown("---")
    st.header("📝 Bước 1: Nhập văn bản bệnh nhân")
    
    # Text input
    user_text = st.text_area(
        "Nhập/dán văn bản về 1 bệnh nhân:",
        height=150,
        placeholder="Ví dụ: Bệnh nhân BN001 là anh Nguyễn Văn An, 45 tuổi, làm kinh doanh. Anh đi từ Hà Nội vào TP.HCM ngày 15/3/2020...",
        key="manual_text_input"
    )
    
    st.session_state.manual_raw_text = user_text
    
    # Step 1: Run NER
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        run_ner_btn = st.button("Chạy NER", type="primary", use_container_width=True, key="manual_run_ner")
    
    if run_ner_btn and user_text.strip():
        with st.spinner("🔄 Đang phân tích văn bản..."):
            try:
                ner_results = predictor.predict(user_text)
                st.session_state.manual_ner_results = ner_results
            except Exception as e:
                st.error(f"❌ Lỗi khi chạy NER: {e}")
    
    # Display NER results
    if st.session_state.manual_ner_results:
        st.markdown("---")
        st.header("📊 Bước 2: Kết quả NER")
        
        with st.expander("Xem chi tiết entities", expanded=True):
            render_entities(user_text, st.session_state.manual_ner_results)
        
        # Step 2: Extract patient info
        st.markdown("---")
        st.header("🔄 Bước 3: Trích xuất thông tin")
        
        col_extract1, col_extract2, col_extract3 = st.columns([1, 1, 2])
        
        with col_extract1:
            extract_btn = st.button("Trích xuất thông tin", type="primary", use_container_width=True, key="manual_extract")
        
        if extract_btn:
            with st.spinner("🔄 Đang xử lý..."):
                try:
                    record = extract_single_patient(
                        st.session_state.manual_ner_results,
                        st.session_state.manual_raw_text
                    )
                    st.session_state.manual_current_record = record
                except Exception as e:
                    st.error(f"❌ Lỗi khi trích xuất: {e}")
    
    # Display extracted record
    if st.session_state.manual_current_record:
        st.markdown("---")
        st.header("👤 Bước 4: Thông tin đã trích xuất")
        
        display_patient_record(st.session_state.manual_current_record)
        
        # Add to list button
        st.markdown("---")
        col_add1, col_add2, col_add3 = st.columns([1, 1, 2])
        
        with col_add1:
            add_btn = st.button("Thêm vào danh sách", type="primary", use_container_width=True, key="manual_add")
        
        with col_add2:
            reset_btn = st.button("Nhập bệnh nhân mới", use_container_width=True, key="manual_reset")
        
        if add_btn:
            st.session_state.manual_patient_list.append(st.session_state.manual_current_record)
            st.session_state.manual_current_record = None
            st.session_state.manual_ner_results = None
            st.session_state.manual_raw_text = ""
            st.rerun()
        
        if reset_btn:
            st.session_state.manual_current_record = None
            st.session_state.manual_ner_results = None
            st.session_state.manual_raw_text = ""
            st.rerun()
    
    # Display patient list
    if st.session_state.manual_patient_list:
        st.markdown("---")
        st.header(f"📋 Danh sách bệnh nhân ({len(st.session_state.manual_patient_list)})")
        
        for idx, record in enumerate(st.session_state.manual_patient_list, 1):
            with st.expander(f"Bệnh nhân #{idx} - {record.patient_id or record.name or 'Unknown'}", expanded=False):
                display_patient_record(record, idx)
        
        # Download CSV
        st.markdown("---")
        col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 2])
        
        with col_dl1:
            csv_data = create_csv_from_records(st.session_state.manual_patient_list)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benh_nhan_manual_{timestamp}.csv"
            
            st.download_button(
                label="Tải xuống CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                type="primary",
                use_container_width=True,
                key="manual_download"
            )


# AUTO MODE TAB
def auto_mode_tab(predictor: NERPredictor):
    """Tab cho chế độ tự động"""
    
    st.markdown("""
    ### Chế độ Tự động với Gemini AI
    
    **Hướng dẫn:** Nhập văn bản nhiều bệnh nhân → Xử lý tự động → Xem kết quả → Tải CSV
    
    Gemini AI tự động tách văn bản, chạy PhoBERT NER và trích xuất thông tin cho tất cả bệnh nhân.
    """)
    
    # Initialize session state for auto mode
    if 'auto_patient_records' not in st.session_state:
        st.session_state.auto_patient_records = []
    if 'auto_text_segments' not in st.session_state:
        st.session_state.auto_text_segments = []
    if 'auto_processing_done' not in st.session_state:
        st.session_state.auto_processing_done = False
    
    # API Key được cung cấp sẵn (built-in)
    GEMINI_API_KEY = "AIzaSyBcwCckmaLuPYatKXKL9n6Pem6XpPIdJuM"
    
    # Sidebar - Stats only (không cần nhập API key nữa)
    with st.sidebar:
        st.markdown("---")
        st.header("📊 Thống kê (Auto)")
        st.metric("Số bệnh nhân", len(st.session_state.auto_patient_records))
        if st.session_state.auto_text_segments:
            st.metric("Số đoạn đã tách", len(st.session_state.auto_text_segments))
        
        if st.button("🗑️ Xóa tất cả (Auto)", key="clear_auto"):
            st.session_state.auto_patient_records = []
            st.session_state.auto_text_segments = []
            st.session_state.auto_processing_done = False
            st.rerun()
    
    st.markdown("---")
    st.header("📝 Nhập văn bản nhiều bệnh nhân")
    
    # Text input
    user_text = st.text_area(
        "Nhập/dán văn bản (có thể chứa thông tin nhiều bệnh nhân):",
        height=200,
        placeholder="Ví dụ:\nBệnh nhân 1 (BN001) là anh Nguyễn Văn An, 45 tuổi...\n\nBệnh nhân 2 tên là chị Trần Thị Bình, 32 tuổi...\n\n...",
        key="auto_text_input"
    )
    
    # Process button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        process_button = st.button("Xử lý tự động", type="primary", use_container_width=True, key="auto_process")
    
    # Process with Gemini
    if process_button:
        if not user_text.strip():
            st.warning("Vui lòng nhập văn bản cần xử lý")
        else:
            # Step 1: Split with Gemini (sử dụng API key built-in)
            with st.spinner("Đang sử dụng Gemini AI để tách văn bản..."):
                try:
                    segments, metadata = split_text_with_gemini(
                        text=user_text,
                        api_key=GEMINI_API_KEY,  # Sử dụng API key đã cấu hình sẵn
                        return_metadata=True
                    )
                    st.session_state.auto_text_segments = segments
                    
                    with st.expander("📊 Thông tin tách văn bản"):
                        st.write(f"- **Số đoạn:** {metadata['num_segments']}")
                        st.write(f"- **Độ dài gốc:** {metadata['original_length']} ký tự")
                        if 'error' in metadata:
                            st.warning(f"⚠️ Có lỗi: {metadata['error']}")
                    
                except Exception as e:
                    st.error(f"❌ Lỗi khi gọi Gemini API: {e}")
                    st.stop()
            
            # Step 2: Process each segment and auto-add to list
            patient_records = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, segment in enumerate(segments):
                progress = (i + 1) / len(segments)
                progress_bar.progress(progress)
                status_text.text(f"Đang xử lý {i+1}/{len(segments)}...")
                
                # Run NER
                ner_results = predictor.predict(segment)
                
                # Extract patient and auto-add to list
                record = extract_single_patient(ner_results, segment)
                patient_records.append(record)
            
            progress_bar.empty()
            status_text.empty()
            
            # Auto-save to session state (tự động thêm tất cả)
            st.session_state.auto_patient_records = patient_records
            st.session_state.auto_processing_done = True
    
    # Display text segments
    if st.session_state.auto_text_segments and st.session_state.auto_processing_done:
        st.markdown("---")
        st.header("📄 Các đoạn văn bản đã tách")
        
        for i, segment in enumerate(st.session_state.auto_text_segments, 1):
            with st.expander(f"Đoạn {i} ({len(segment)} ký tự)"):
                st.text(segment)
    
    # Display patient records
    if st.session_state.auto_patient_records:
        st.markdown("---")
        st.header(f"👥 Danh sách Bệnh nhân ({len(st.session_state.auto_patient_records)})")
        
        # Tạo list để theo dõi bệnh nhân cần xóa
        patients_to_remove = []
        
        for idx, record in enumerate(st.session_state.auto_patient_records, 1):
            col_expand, col_remove = st.columns([10, 1])
            
            with col_expand:
                with st.expander(f"Bệnh nhân #{idx} - {record.patient_id or record.name or 'Unknown'}", expanded=False):
                    display_patient_record(record, idx)
            
            with col_remove:
                if st.button("🗑️", key=f"remove_auto_{idx}", help="Xóa bệnh nhân này"):
                    patients_to_remove.append(idx - 1)  # index 0-based
        
        # Xử lý xóa bệnh nhân
        if patients_to_remove:
            for idx in sorted(patients_to_remove, reverse=True):
                st.session_state.auto_patient_records.pop(idx)
            st.rerun()
        
        # Download CSV
        st.markdown("---")
        st.subheader("📥 Export dữ liệu")
        
        col_download1, col_download2, col_download3 = st.columns([1, 1, 2])
        
        with col_download1:
            csv_data = create_csv_from_records(st.session_state.auto_patient_records)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benh_nhan_auto_{timestamp}.csv"
            
            st.download_button(
                label=f"Tải xuống CSV ({len(st.session_state.auto_patient_records)} BN)",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                type="primary",
                use_container_width=True,
                key="auto_download"
            )
        
        with col_download2:
            if st.button("Xem trước CSV", use_container_width=True, key="auto_preview"):
                df = pd.DataFrame([r.to_dict() for r in st.session_state.auto_patient_records])
                st.dataframe(df, use_container_width=True)


# MAIN APP
def main():
    """Main app function"""
    
    # Page config
    st.set_page_config(
        page_title="NER Patient Extraction",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🏥 Hệ thống Trích xuất Thông tin Bệnh nhân COVID-19")
    st.markdown("""
    **Ứng dụng NER (Named Entity Recognition)** cho việc trích xuất thông tin bệnh nhân từ văn bản tiếng Việt.
    
    Chọn chế độ phù hợp:
    - **Manual Mode**: Xử lý từng bệnh nhân một cách thủ công (phù hợp 1-2 bệnh nhân)
    - **Auto Mode**: Tự động tách và xử lý nhiều bệnh nhân (cần Gemini API)
    """)
    
    # Load model (shared)
    predictor = load_ner_model()
    
    if predictor is None:
        st.error("❌ Không thể tải mô hình. Vui lòng kiểm tra cấu hình.")
        st.stop()
    
    st.markdown("---")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Manual Mode", "Auto Mode"])
    
    with tab1:
        manual_mode_tab(predictor)
    
    with tab2:
        auto_mode_tab(predictor)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Phát triển bởi NER COVID-19 Team | Powered by PhoBERT & Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
