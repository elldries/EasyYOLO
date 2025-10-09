#!/usr/bin/env python3 ОПИСАНИЕ ФУНКЦИЙ ПИСАЛА ИИ, НЕ ЕБИТЕ МОЗГИ, ЭТО ДЛЯ ВАШЕГО ЖЕ БЛАГА! THE AI WROTE THE FUNCTION DESCRIPTIONS, DON'T FUCK WITH MY HEAD, IT'S FOR YOUR OWN GOOD!

"""
Lieris Installer
"""

import subprocess
import sys
import platform
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Installing: {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"✓ {description} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing {description}")
        print(f"Error: {e.stderr}")
        return False

def check_cuda():
    """Check if CUDA is available"""
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("\n✓ NVIDIA GPU detected - CUDA support will be enabled")
            return True
    except:
        pass
    print("\n⚠ No NVIDIA GPU detected - CPU-only mode will be used")
    return False

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║           Lieris   - Dependency Installer                ║
║   by: elldries                                           ║
║  This script will install all required dependencies:     ║
║  - PyTorch (with CUDA if available)                      ║
║  - Ultralytics YOLOv8/v11                                ║
║  - OpenCV                                                ║
║  - ONNX Runtime                                          ║
║  - Additional utilities                                  ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check CUDA availability
    has_cuda = check_cuda()
    
    # Upgrade pip
    run_command(f"{sys.executable} -m pip install --upgrade pip", "pip upgrade")
    
    # Install PyTorch
    if has_cuda:
        pytorch_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    else:
        pytorch_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio"
    
    if not run_command(pytorch_cmd, "PyTorch"):
        print("\n⚠ PyTorch installation failed. Trying alternative method...")
        run_command(f"{sys.executable} -m pip install torch torchvision torchaudio", "PyTorch (CPU)")
    
    # Core AI packages
    packages = {
        "Ultralytics YOLO": "ultralytics",
        "OpenCV": "opencv-python",
        "Pillow": "Pillow",
        "NumPy": "numpy",
        "ONNX": "onnx",
        "ONNX Runtime": "onnxruntime" if not has_cuda else "onnxruntime-gpu",
        "ONNX Simplifier": "onnx-simplifier",
        "PyYAML": "PyYAML",
        "tqdm": "tqdm",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn",
    }
    
    for name, package in packages.items():
        run_command(f"{sys.executable} -m pip install {package}", name)
    
    # Verify installation
    print(f"\n{'='*60}")
    print("Verifying installations...")
    print(f"{'='*60}")
    
    verification_failed = False
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  CUDA not available (CPU mode)")
    except ImportError:
        print("✗ PyTorch not found")
        verification_failed = True
    
    try:
        import ultralytics
        from ultralytics import YOLO
        print(f"✓ Ultralytics YOLO {ultralytics.__version__}")
    except ImportError:
        print("✗ Ultralytics not found")
        verification_failed = True
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not found")
        verification_failed = True
    
    try:
        import onnx
        import onnxruntime
        print(f"✓ ONNX {onnx.__version__}")
        print(f"✓ ONNX Runtime {onnxruntime.__version__}")
    except ImportError:
        print("✗ ONNX packages not found")
        verification_failed = True
    
    if verification_failed:
        print("\n⚠ Some packages failed to install. Please check the errors above.")
        return 1
    
    print(f"\n{'='*60}")
    print("✓ All dependencies installed successfully!")
    print(f"{'='*60}")
    print("\nYou can now:")
    print("1. Train models using: python Train.py")
    print("2. Run Lieris application")
    print(f"{'='*60}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
