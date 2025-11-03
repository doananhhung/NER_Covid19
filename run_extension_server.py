# run_extension_server.py
"""
Script khởi động Backend API Server cho Chrome Extension
"""

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def check_dependencies():
    """Kiểm tra các dependencies cần thiết"""
    print("Kiểm tra dependencies...")
    
    missing = []
    
    try:
        import fastapi
    except ImportError:
        missing.append("fastapi")
    
    try:
        import uvicorn
    except ImportError:
        missing.append("uvicorn")
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    if missing:
        print("\nThieu cac package sau:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nChay lenh: pip install -r requirements.txt")
        print("Va: pip install -r backend_api/requirements_api.txt")
        return False
    
    print("Tat ca dependencies da san sang!")
    return True


def check_model():
    """Kiểm tra model có tồn tại không"""
    print("\nKiem tra model...")
    
    from src import config as ner_config
    model_path = ner_config.MODEL_OUTPUT_DIR
    
    if not os.path.exists(model_path):
        print(f"\nLoi: Khong tim thay model tai {model_path}")
        print("Vui long train model truoc hoac tai model da train.")
        return False
    
    # Check các file cần thiết
    required_files = ['config.json', 'pytorch_model.bin', 'vocab.txt']
    safetensors_file = 'model.safetensors'
    
    has_model = False
    for req_file in required_files:
        if req_file == 'pytorch_model.bin':
            # Có thể là pytorch_model.bin hoặc model.safetensors
            if os.path.exists(os.path.join(model_path, req_file)):
                has_model = True
            elif os.path.exists(os.path.join(model_path, safetensors_file)):
                has_model = True
        else:
            if not os.path.exists(os.path.join(model_path, req_file)):
                print(f"Canh bao: Thieu file {req_file}")
    
    if not has_model:
        print("Loi: Khong tim thay file model weights")
        return False
    
    print(f"Model co tai: {model_path}")
    return True


def check_vncorenlp():
    """Kiểm tra VnCoreNLP"""
    print("\nKiem tra VnCoreNLP...")
    
    vncorenlp_path = os.path.join(PROJECT_ROOT, 'vncorenlp_models')
    
    if not os.path.exists(vncorenlp_path):
        print(f"Canh bao: VnCoreNLP chua duoc setup tai {vncorenlp_path}")
        print("Chay: python setup_vncorenlp.py")
        print("Server se khoi dong nhung khong co word segmentation.")
        return False
    
    print("VnCoreNLP san sang!")
    return True


def main():
    """Main function"""
    print("=" * 80)
    print(" KHOI DONG BACKEND SERVER CHO CHROME EXTENSION")
    print("=" * 80)
    print()
    
    # Checks
    if not check_dependencies():
        sys.exit(1)
    
    if not check_model():
        sys.exit(1)
    
    check_vncorenlp()  # Warning only
    
    # Start server
    print("\n" + "=" * 80)
    print(" BAT DAU KHOI DONG SERVER...")
    print("=" * 80)
    print()
    print("Server se chay tai: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print()
    print("Nhan Ctrl+C de dung server")
    print()
    
    try:
        import uvicorn
        from backend_api.main import app
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    
    except KeyboardInterrupt:
        print("\n\nDang dung server...")
        print("Server da dung!")
    
    except Exception as e:
        print(f"\n\nLoi khi khoi dong server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
