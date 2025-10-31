# test_text_processor.py
#
# Script để test text processor và word segmentation.

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from text_processor import get_text_processor


def test_segmentation():
    """
    Test chức năng tách từ tiếng Việt.
    """
    print("Kiểm tra Text Processor...")
    print("=" * 80)
    
    processor = get_text_processor()
    
    if not processor.is_available():
        print("\nVnCoreNLP chưa được cài đặt hoặc chưa sẵn sàng.")
        print("\nĐể cài đặt VnCoreNLP models, chạy:")
        print("  python setup_vncorenlp.py")
        return
    
    print("\nVnCoreNLP đã sẵn sàng!")
    print("\nThử nghiệm tách từ:")
    print("-" * 80)
    
    test_sentences = [
        "Bệnh nhân được đưa đi bệnh viện",
        "Hà Nội ghi nhận 15 ca mắc COVID-19 mới",
        "Bộ Y tế khuyến cáo người dân đeo khẩu trang",
        "Người bệnh có triệu chứng ho và sốt cao"
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\nCâu {i}:")
        print(f"  Gốc:      {sentence}")
        segmented = processor.segment_text(sentence)
        print(f"  Đã tách:  {segmented}")
    
    print("\n" + "=" * 80)
    print("Hoàn thành kiểm tra!")


if __name__ == "__main__":
    test_segmentation()
