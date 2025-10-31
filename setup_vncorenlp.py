# setup_vncorenlp.py
#
# Script này giúp tải xuống và cài đặt VnCoreNLP models.

import os
import sys
import urllib.request
import zipfile
import shutil

# Thêm thư mục gốc vào Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src import config


def download_file(url: str, dest_path: str) -> None:
    """
    Tải file từ URL về dest_path với progress bar.
    """
    print(f"Đang tải từ: {url}")
    
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r  Tiến trình: {percent}%")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, dest_path, reporthook)
    print()  # New line sau progress bar


def setup_vncorenlp():
    """
    Tải xuống và cài đặt VnCoreNLP models.
    """
    try:
        print("Đang kiểm tra VnCoreNLP models...")
        
        models_subdir = os.path.join(config.VNCORENLP_MODELS_DIR, 'models')
        
        if os.path.exists(models_subdir):
            print(f"VnCoreNLP models đã tồn tại tại: {config.VNCORENLP_MODELS_DIR}")
            
            response = input("Bạn có muốn tải lại không? (y/n): ")
            if response.lower() != 'y':
                print("Hủy cài đặt.")
                return
            
            print(f"Đang xóa thư mục cũ...")
            shutil.rmtree(config.VNCORENLP_MODELS_DIR)
        
        print(f"Tạo thư mục: {config.VNCORENLP_MODELS_DIR}")
        os.makedirs(config.VNCORENLP_MODELS_DIR, exist_ok=True)
        os.makedirs(models_subdir, exist_ok=True)
        
        print(f"\nĐang tải VnCoreNLP models...")
        print("Quá trình này có thể mất vài phút...\n")
        
        # URLs của VnCoreNLP models từ GitHub releases
        vncorenlp_url = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.2.jar"
        
        # Models được lưu trữ ở Google Drive hoặc GitHub releases
        # Sử dụng URL từ py_vncorenlp source code
        models_urls = {
            'wordsegmenter': 'https://github.com/vncorenlp/VnCoreNLP/raw/master/models/wordsegmenter/vi-vocab',
            'wordsegmenter_rdr': 'https://github.com/vncorenlp/VnCoreNLP/raw/master/models/wordsegmenter/wordsegmenter.rdr',
        }
        
        jar_path = os.path.join(config.VNCORENLP_MODELS_DIR, 'VnCoreNLP-1.2.jar')
        
        # Tải JAR file
        print("1. Đang tải VnCoreNLP JAR file...")
        download_file(vncorenlp_url, jar_path)
        print(f"   ✓ Đã lưu: {jar_path}")
        
        # Tạo thư mục cho wordsegmenter
        wordseg_dir = os.path.join(models_subdir, 'wordsegmenter')
        os.makedirs(wordseg_dir, exist_ok=True)
        
        # Tải models
        print("\n2. Đang tải word segmentation models...")
        for name, url in models_urls.items():
            filename = os.path.basename(url)
            dest_path = os.path.join(wordseg_dir, filename)
            print(f"   - Đang tải {filename}...")
            download_file(url, dest_path)
            print(f"     ✓ Đã lưu: {dest_path}")
        
        print("\n" + "="*80)
        print("Cài đặt thành công!")
        print(f"VnCoreNLP models đã được lưu tại: {config.VNCORENLP_MODELS_DIR}")
        print("="*80)
        
    except ImportError as e:
        print(f"Lỗi import: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nLỗi khi tải VnCoreNLP models: {e}")
        import traceback
        traceback.print_exc()
        print("\nGợi ý: Bạn có thể thử sử dụng py_vncorenlp trực tiếp:")
        print("  import py_vncorenlp")
        print(f"  py_vncorenlp.download_model(save_dir='{config.VNCORENLP_MODELS_DIR}')")
        print("\nLưu ý: Cần cài đặt wget trên Windows hoặc dùng WSL")
        sys.exit(1)


if __name__ == "__main__":
    setup_vncorenlp()
