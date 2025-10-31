#!/usr/bin/env python
# run_app.py
# Script wrapper để chạy Streamlit app một cách đúng đắn
# Tự động phát hiện và sử dụng venv nếu có, nếu không thì dùng Python hệ thống

import os
import sys
import subprocess

# Đảm bảo working directory là thư mục gốc của project
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

# Kiểm tra và chọn Python interpreter
venv_python = os.path.join(project_root, '.venv', 'Scripts', 'python.exe')

if os.path.exists(venv_python):
    python_exe = venv_python
    print(" Sử dụng Python từ virtual environment (.venv)")
else:
    python_exe = sys.executable  # Dùng Python hiện tại (hệ thống)
    print("  Virtual environment không tìm thấy")
    print(" Sử dụng Python hệ thống")
    print(" Khuyến nghị: Tạo venv bằng lệnh: python -m venv .venv")

app_path = os.path.join(project_root, 'app', 'app.py')

# Hiển thị thông tin
print(f" Python: {python_exe}")
print(f" Working directory: {os.getcwd()}")
print(f" App path: {app_path}")
print()

# Chạy Streamlit
cmd = [python_exe, '-m', 'streamlit', 'run', app_path]

try:
    subprocess.run(cmd)
except KeyboardInterrupt:
    print("\n\n Đã dừng ứng dụng!")
except Exception as e:
    print(f"\n Lỗi: {e}")
    print("\n Hãy đảm bảo:")
    print("   1. Đã cài đặt streamlit: pip install streamlit")
    print("   2. Đã cài đặt tất cả packages: pip install -r requirements.txt")
