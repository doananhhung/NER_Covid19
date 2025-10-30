# app/utils.py
# Chứa các hàm tiện ích cho ứng dụng Streamlit.

import streamlit as st
import re

# Định nghĩa một bảng màu để tô màu các loại thực thể khác nhau
# Bạn có thể tùy chỉnh thêm màu sắc tại đây
TAG_COLORS = {
    "PATIENT_ID": "#f9c5c7",
    "SYMPTOM_AND_DISEASE": "#b2d8d8",
    "LOCATION": "#ace4aa",
    "DATE": "#ffb347",
    "ORGANIZATION": "#c7b5e4",
    "AGE": "#fdfd96",
    "GENDER": "#87ceeb",
    "NAME": "#ffcccb",
    "TRANSPORTATION": "#d3d3d3",
    "JOB": "#f5deb3",
    "DEFAULT": "#f0f2f6"  # Màu mặc định
}

def render_entities(sentence, entities):
    """
    Hiển thị câu với các thực thể được tô màu.

    Args:
        sentence (str): Câu văn bản gốc.
        entities (list): Danh sách các thực thể được mô hình dự đoán (có thể có start/end positions).
    """
    if not entities:
        st.write(sentence)
        return

    # Kiểm tra xem entities đã có thông tin vị trí chưa
    has_positions = all('start' in e and 'end' in e and e['start'] != -1 for e in entities)
    
    if has_positions:
        # Sử dụng vị trí có sẵn
        entities_with_positions = [
            {
                'text': e['text'],
                'tag': e['tag'],
                'start': e['start'],
                'end': e['end']
            }
            for e in entities if e['start'] != -1
        ]
    else:
        # Tìm vị trí thực tế của các entities trong câu gốc (logic cũ)
        entities_with_positions = []
        
        for entity in entities:
            entity_text = entity['text'].strip()
            entity_tag = entity['tag']
            
            if not entity_text:
                continue
            
            # Tìm tất cả các vị trí có thể xuất hiện của entity trong câu
            # Sử dụng regex để tìm kiếm linh hoạt hơn (cho phép khoảng trắng)
            # Escape các ký tự đặc biệt trong regex
            escaped_text = re.escape(entity_text)
            # Thay thế khoảng trắng bằng pattern cho phép nhiều khoảng trắng
            pattern = escaped_text.replace(r'\ ', r'\s+')
            
            matches = list(re.finditer(pattern, sentence))
            
            if matches:
                # Lấy match đầu tiên chưa được sử dụng
                for match in matches:
                    start_idx = match.start()
                    end_idx = match.end()
                    matched_text = sentence[start_idx:end_idx]
                    
                    # Kiểm tra xem vị trí này có bị overlap với entity khác không
                    is_overlap = False
                    for existing_entity in entities_with_positions:
                        if not (end_idx <= existing_entity['start'] or start_idx >= existing_entity['end']):
                            is_overlap = True
                            break
                    
                    if not is_overlap:
                        entities_with_positions.append({
                            'text': matched_text,
                            'tag': entity_tag,
                            'start': start_idx,
                            'end': end_idx
                        })
                        break
    
    # Sắp xếp các thực thể theo vị trí bắt đầu
    entities_with_positions.sort(key=lambda e: e['start'])
    
    # Xây dựng HTML
    rendered_html = ""
    last_idx = 0

    for entity in entities_with_positions:
        start_idx = entity['start']
        end_idx = entity['end']
        entity_text = entity['text']
        entity_tag = entity['tag']
        
        # Thêm phần văn bản không phải thực thể (escape HTML)
        text_before = sentence[last_idx:start_idx]
        rendered_html += text_before
        
        # Thêm phần thực thể đã được tô màu
        color = TAG_COLORS.get(entity_tag, TAG_COLORS["DEFAULT"])
        rendered_html += (
            f'<mark style="background-color: {color}; padding: 2px 4px; '
            f'margin: 0 2px; border-radius: 3px; '
            f'font-weight: normal; display: inline-block;">'
            f'{entity_text}'
            f' <span style="color: #555; font-size: 0.75em; font-weight: bold;">({entity_tag})</span>'
            f'</mark>'
        )
        
        last_idx = end_idx
    
    # Thêm phần văn bản còn lại sau thực thể cuối cùng
    rendered_html += sentence[last_idx:]
    
    # Hiển thị với unsafe_allow_html
    st.markdown(rendered_html, unsafe_allow_html=True)
