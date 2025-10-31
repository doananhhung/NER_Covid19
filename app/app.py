# app/app.py
# File chính để chạy ứng dụng web demo bằng Streamlit.

import sys
import os
import streamlit as st

# Thêm thư mục `src` vào Python path để có thể import các module
# Sử dụng __file__ của chính file này để xác định đúng PROJECT_ROOT
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_FILE_DIR)  # Lùi về thư mục gốc
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)  # insert(0) để ưu tiên tìm trong project

# Đổi working directory về PROJECT_ROOT để tránh path confusion
os.chdir(PROJECT_ROOT)

from src.inference import NERPredictor
from src import config as ner_config

# Import utils từ cùng thư mục app
import sys
sys.path.insert(0, CURRENT_FILE_DIR)
from utils import render_entities

@st.cache_resource
def load_ner_model():
    """
    Tải mô hình NER. Sử dụng cache của Streamlit để chỉ tải một lần.
    Returns:
        NERPredictor: Một instance của lớp NERPredictor, hoặc None nếu có lỗi.
    """
    try:
        print("=" * 80)
        print("DEBUG INFO:")
        print(f"Current working directory: {os.getcwd()}")
        print(f"PROJECT_ROOT: {PROJECT_ROOT}")
        print(f"Model path: {ner_config.MODEL_OUTPUT_DIR}")
        print(f"VnCoreNLP path: {ner_config.VNCORENLP_MODELS_DIR}")
        print("=" * 80)
        
        print("Đang tải mô hình NER...")
        # BẬT word segmentation để cải thiện độ chính xác
        predictor = NERPredictor(
            model_path=ner_config.MODEL_OUTPUT_DIR,
            use_word_segmentation=True
        )
        # Kiểm tra xem model có thực sự được tải không
        if predictor.model is None:
            return None
        print("Tải mô hình thành công.")
        print(f"Word segmentation: {'Enabled ' if predictor.use_word_segmentation else 'Disabled '}")
        return predictor
    except Exception as e:
        print(f"LỖI NGHIÊM TRỌNG khi tải mô hình: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Hàm chính chạy ứng dụng Streamlit."""
    # --- Cấu hình trang ---
    st.set_page_config(
        page_title="Nhận dạng Thực thể Y tế",
        page_icon="🩺",
        layout="wide"
    )

    # --- Giao diện người dùng ---
    st.title("🩺 Hệ thống Nhận dạng Thực thể Y tế (NER)")
    st.markdown(
        "Nhập một câu văn bản liên quan đến y tế (COVID-19) vào ô bên dưới "
        "và nhấn **'Phân tích'** để xem các thực thể được mô hình nhận dạng."
    )
    
    # --- Tải mô hình ---
    predictor = load_ner_model()

    # --- Kiểm tra nếu model tải thất bại ---
    if predictor is None:
        st.error(
            f"**Lỗi nghiêm trọng khi tải mô hình!**\n\n"
            f"Không thể tải mô hình từ đường dẫn: `{os.path.join(PROJECT_ROOT, ner_config.MODEL_OUTPUT_DIR)}`.\n\n"
            "**Vui lòng kiểm tra:**\n"
            "1. Bạn đã chạy script `train.py` thành công chưa?\n"
            "2. Thư mục `models/phobert-ner-covid` có tồn tại và chứa đủ các file cần thiết không?\n"
            "3. Xem lại cửa sổ Terminal để biết thông báo lỗi chi tiết."
        )
        # Dừng ứng dụng tại đây
        return

    # Ô nhập liệu
    default_text = "Bệnh nhân nữ 35 tuổi, mã số BN2345, quê ở Hà Nội, nhập viện ngày 15/08/2021 với triệu chứng ho và sốt."
    user_input = st.text_area("Nhập văn bản của bạn:", default_text, height=200)
    
    # Thêm thông tin về word segmentation
    if predictor.use_word_segmentation:
        st.success(" Word Segmentation: **Đã bật** (để cải thiện độ chính xác)")
    else:
        st.warning(" Word Segmentation: **Chưa bật** (có thể ảnh hưởng độ chính xác)")
    
   
    # Nút bấm
    if st.button("Phân tích", type="primary"):
        if user_input.strip():
            # Hiển thị thông tin về độ dài văn bản
            char_count = len(user_input)
            word_count = len(user_input.split())
            st.caption(f" Độ dài: {char_count} ký tự, ~{word_count} từ")
            
            with st.spinner(" Đang phân tích văn bản..."):
                try:
                    # Redirect output để capture progress messages
                    import io
                    from contextlib import redirect_stdout, redirect_stderr
                    
                    # Capture output
                    stdout_capture = io.StringIO()
                    stderr_capture = io.StringIO()
                    
                    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                        entities = predictor.predict(user_input, show_debug=True)
                    
                    # Hiển thị progress messages
                    progress_output = stdout_capture.getvalue()
                    error_output = stderr_capture.getvalue()
                    
                    if progress_output.strip() or error_output.strip():
                        with st.expander(" Chi tiết xử lý", expanded=False):
                            if progress_output.strip():
                                st.text("STDOUT:")
                                st.text(progress_output)
                            if error_output.strip():
                                st.text("STDERR:")
                                st.text(error_output)
                    
                    # Kiểm tra kết quả
                    if entities is None:
                        st.error(" Lỗi: Không thể xử lý văn bản. Vui lòng kiểm tra log.")
                        return
                    
                    if len(entities) == 0:
                        st.warning(" Không tìm thấy thực thể y tế nào trong văn bản.")
                        st.info(" **Gợi ý:**\n"
                               "- Thử với văn bản có chứa tên bệnh viện, tên bệnh nhân, địa điểm\n"
                               "- Ví dụ: 'Bệnh nhân Nguyễn Văn A, 35 tuổi, tại Bệnh viện Bạch Mai'")
                    else:
                        st.success(f" Hoàn tất! Tìm thấy {len(entities)} thực thể.")
                        
                        st.subheader("Kết quả Nhận dạng:")
                        render_entities(user_input, entities)
                        
                        with st.expander("Xem kết quả dạng JSON"):
                            st.json(entities)
                            
                except Exception as e:
                    st.error(f" Lỗi khi phân tích văn bản: {e}")
                    import traceback
                    with st.expander("Chi tiết lỗi"):
                        st.code(traceback.format_exc())
        else:
            st.warning("Vui lòng nhập văn bản để phân tích.")

if __name__ == "__main__":
    main()

