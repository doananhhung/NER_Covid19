# src/text_processor.py
#
# Module này cung cấp các tiện ích để xử lý văn bản tiếng Việt,
# bao gồm tách từ (word segmentation) sử dụng VnCoreNLP.

import os
from typing import Optional
import sys

# Thêm thư mục src vào sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config


class VietnameseTextProcessor:
    """
    Lớp xử lý văn bản tiếng Việt với chức năng tách từ (word segmentation).
    Sử dụng py_vncorenlp để tách từ tiếng Việt.
    """
    
    _instance = None
    
    def __new__(cls):
        """
        Singleton pattern để tránh tải VnCoreNLP nhiều lần.
        """
        if cls._instance is None:
            cls._instance = super(VietnameseTextProcessor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """
        Khởi tạo text processor với VnCoreNLP.
        """
        if self._initialized:
            return
            
        self.word_segmenter = None
        self._initialized = True
        self._initialize_segmenter()
    
    def _initialize_segmenter(self) -> None:
        """
        Khởi tạo VnCoreNLP word segmenter.
        """
        try:
            from py_vncorenlp import VnCoreNLP
            
            models_subdir = os.path.join(config.VNCORENLP_MODELS_DIR, 'models')
            
            if not os.path.exists(models_subdir):
                print(f"VnCoreNLP models không tìm thấy tại: {models_subdir}")
                print("\nĐể cài đặt VnCoreNLP models, chạy lệnh sau:")
                print("  python setup_vncorenlp.py")
                print("\nHoặc trong Python:")
                print("  import py_vncorenlp")
                print("  py_vncorenlp.download_model(save_dir='./vncorenlp_models')")
                return
            
            print(f"Đang khởi tạo VnCoreNLP từ: {config.VNCORENLP_MODELS_DIR}")
            self.word_segmenter = VnCoreNLP(
                save_dir=config.VNCORENLP_MODELS_DIR,
                annotators=config.VNCORENLP_ANNOTATORS
            )
            print("VnCoreNLP đã được khởi tạo thành công")
            
        except ImportError:
            print("Cảnh báo: py_vncorenlp chưa được cài đặt")
            print("Cài đặt bằng lệnh: pip install py_vncorenlp")
        except Exception as e:
            print(f"Lỗi khi khởi tạo VnCoreNLP: {e}")
    
    def segment_text(self, text: str) -> Optional[str]:
        """
        Tách từ tiếng Việt sử dụng VnCoreNLP.
        
        Args:
            text (str): Văn bản đầu vào.
            
        Returns:
            Optional[str]: Văn bản đã được tách từ (các từ ghép nối bằng dấu _).
                          Trả về None nếu VnCoreNLP không khả dụng.
                          
        Examples:
            >>> processor = VietnameseTextProcessor()
            >>> text = "Bệnh nhân được đưa đi bệnh viện"
            >>> segmented = processor.segment_text(text)
            >>> print(segmented)
            'Bệnh_nhân được đưa đi bệnh_viện'
        """
        if not self.word_segmenter:
            print("Cảnh báo: VnCoreNLP không khả dụng. Trả về văn bản gốc.")
            return text
        
        try:
            segmented_result = self.word_segmenter.word_segment(text)
            
            if not segmented_result:
                return text
            
            if isinstance(segmented_result, list):
                if len(segmented_result) > 0 and isinstance(segmented_result[0], list):
                    result = ' '.join([' '.join(sent) for sent in segmented_result])
                else:
                    result = ' '.join(segmented_result)
                return result
            
            return text
            
        except Exception as e:
            print(f"Lỗi khi tách từ: {e}")
            return text
    
    def is_available(self) -> bool:
        """
        Kiểm tra xem VnCoreNLP có sẵn sàng không.
        
        Returns:
            bool: True nếu VnCoreNLP đã được khởi tạo thành công.
        """
        return self.word_segmenter is not None


def get_text_processor() -> VietnameseTextProcessor:
    """
    Hàm tiện ích để lấy instance của VietnameseTextProcessor.
    
    Returns:
        VietnameseTextProcessor: Instance duy nhất của text processor.
    """
    return VietnameseTextProcessor()


def segment_vietnamese_text(text: str) -> Optional[str]:
    """
    Hàm tiện ích để tách từ tiếng Việt.
    
    Args:
        text (str): Văn bản đầu vào.
        
    Returns:
        Optional[str]: Văn bản đã được tách từ.
    """
    processor = get_text_processor()
    return processor.segment_text(text)
