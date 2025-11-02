# src/patient_extraction/gemini_splitter.py
"""
Module sử dụng Gemini API để tách văn bản thành các đoạn,
mỗi đoạn tương ứng với thông tin của 1 bệnh nhân.
"""

from typing import List, Optional, Tuple
import re


class GeminiTextSplitter:
    """
    Class sử dụng Gemini API để tách văn bản về nhiều bệnh nhân
    thành các đoạn văn bản riêng biệt.
    """
    
    def __init__(self, api_key: str):
        """
        Khởi tạo splitter với API key
        
        Args:
            api_key: Google Gemini API key
        """
        self.api_key = api_key
        self.client = None
        self.model_name = "gemini-2.5-flash"
        self._initialize_client()
    
    def _initialize_client(self):
        """Khởi tạo Gemini client"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai
            print("Gemini API client initialized successfully")
        except ImportError as ie:
            print(f"Lỗi: Chưa cài đặt google-generativeai. Chạy: pip install google-generativeai")
            print(f"   Chi tiết: {ie}")
            self.client = None
        except Exception as e:
            print(f"Lỗi khởi tạo Gemini client: {e}")
            import traceback
            traceback.print_exc()
            self.client = None
    
    def is_available(self) -> bool:
        """Kiểm tra xem Gemini API có sẵn sàng không"""
        return self.client is not None
    
    def split_text_by_patients(
        self, 
        text: str,
        return_metadata: bool = False
    ) -> List[str] | Tuple[List[str], dict]:
        """
        Tách văn bản thành các đoạn, mỗi đoạn cho 1 bệnh nhân
        
        Args:
            text: Văn bản đầu vào (có thể chứa thông tin nhiều bệnh nhân)
            return_metadata: Có trả về metadata không
            
        Returns:
            List[str]: Danh sách các đoạn văn bản (mỗi đoạn = 1 bệnh nhân)
            hoặc Tuple[List[str], dict] nếu return_metadata=True
        """
        if not self.is_available():
            raise RuntimeError("Gemini API không khả dụng. Kiểm tra API key và cài đặt.")
        
        # Tạo prompt cho Gemini
        prompt = self._create_splitting_prompt(text)
        
        try:
            # Gọi API
            model = self.client.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            
            # Parse response
            segments = self._parse_response(response.text)
            
            # Validate segments
            if not segments:
                print("Gemini không tách được văn bản. Trả về toàn bộ văn bản gốc.")
                segments = [text]
            
            metadata = {
                'original_length': len(text),
                'num_segments': len(segments),
                'api_response': response.text[:500]  # Lưu 500 ký tự đầu
            }
            
            if return_metadata:
                return segments, metadata
            return segments
            
        except Exception as e:
            print(f" Lỗi khi gọi Gemini API: {e}")
            # Fallback: trả về văn bản gốc
            if return_metadata:
                return [text], {'error': str(e)}
            return [text]
    
    def _create_splitting_prompt(self, text: str) -> str:
        """
        Tạo prompt cho Gemini để tách văn bản
        
        Args:
            text: Văn bản cần tách
            
        Returns:
            str: Prompt
        """
        prompt = f"""
Nhiệm vụ: Phân tích văn bản sau và tách thành các đoạn văn bản riêng biệt, trong đó MỖI ĐOẠN chỉ chứa thông tin về MỘT bệnh nhân duy nhất.

Hướng dẫn:
1. Đọc kỹ văn bản và xác định có bao nhiêu bệnh nhân được nhắc đến
2. Tách văn bản thành các đoạn, mỗi đoạn chỉ nói về 1 bệnh nhân
3. Mỗi đoạn nên bao gồm TẤT CẢ thông tin liên quan đến bệnh nhân đó (ID, tên, tuổi, giới tính, địa điểm, ngày tháng, triệu chứng, v.v.)
4. KHÔNG bịa thêm thông tin, chỉ trích xuất từ văn bản gốc
5. GIỮ NGUYÊN các con số, tên riêng, địa điểm, ngày tháng

Format trả về:
---PATIENT_1---
[Toàn bộ thông tin của bệnh nhân 1]
---END---

---PATIENT_2---
[Toàn bộ thông tin của bệnh nhân 2]
---END---

(Tiếp tục cho các bệnh nhân khác nếu có)

Văn bản cần phân tích:
{text}

Kết quả (chỉ trả về các đoạn đã tách, không giải thích):
"""
        return prompt
    
    def _parse_response(self, response_text: str) -> List[str]:
        """
        Parse response từ Gemini để lấy các đoạn văn bản
        
        Args:
            response_text: Response từ Gemini API
            
        Returns:
            List[str]: Danh sách các đoạn văn bản
        """
        # Pattern để tìm các đoạn PATIENT
        pattern = r'---PATIENT_\d+---(.*?)---END---'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        if matches:
            # Làm sạch các đoạn
            segments = [match.strip() for match in matches if match.strip()]
            return segments
        
        # Nếu không match được pattern, thử tách theo số thứ tự
        # Ví dụ: "Bệnh nhân 1:", "Bệnh nhân 2:", etc.
        lines = response_text.split('\n')
        segments = []
        current_segment = []
        
        for line in lines:
            # Check nếu line bắt đầu segment mới
            if re.match(r'^(Bệnh nhân|Patient|BN)\s*\d+', line, re.IGNORECASE):
                if current_segment:
                    segments.append('\n'.join(current_segment).strip())
                    current_segment = []
            current_segment.append(line)
        
        # Thêm segment cuối
        if current_segment:
            segments.append('\n'.join(current_segment).strip())
        
        return [s for s in segments if s]  # Loại bỏ empty strings


def split_text_with_gemini(
    text: str, 
    api_key: str,
    return_metadata: bool = False
) -> List[str] | Tuple[List[str], dict]:
    """
    Helper function để tách văn bản sử dụng Gemini API
    
    Args:
        text: Văn bản cần tách
        api_key: Gemini API key
        return_metadata: Có trả về metadata không
        
    Returns:
        List[str]: Danh sách các đoạn văn bản
        hoặc Tuple[List[str], dict] nếu return_metadata=True
    """
    splitter = GeminiTextSplitter(api_key=api_key)
    return splitter.split_text_by_patients(text, return_metadata=return_metadata)
