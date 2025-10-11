@echo off
setlocal enabledelayedexpansion

echo 
echo Lieris
echo 
echo.

REM Check if Python is installed
echo [1/7] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo IMPORTANT: During installation, check the box "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%
echo.

REM Check Python version (requires 3.8+)
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)
if %MAJOR% LSS 3 (
    echo ERROR: Python 3.8 or higher is required!
    echo Current version: %PYTHON_VERSION%
    echo.
    pause
    exit /b 1
)
if %MAJOR% EQU 3 if %MINOR% LSS 8 (
    echo ERROR: Python 3.8 or higher is required!
    echo Current version: %PYTHON_VERSION%
    echo.
    pause
    exit /b 1
)

echo [2/7] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip, continuing anyway...
)
echo.

echo [3/7] Detecting CUDA and GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo No NVIDIA GPU detected - Installing CPU version
    echo.
    set PYTORCH_INDEX=https://download.pytorch.org/whl/cpu
    set PYTORCH_TYPE=CPU
) else (
    echo NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo.
    
    REM Check if Python 3.13+ is being used
    if %MAJOR% EQU 3 if %MINOR% GEQ 13 (
        echo Python 3.13 detected - Using CUDA 11.8 for compatibility
        set PYTORCH_INDEX=https://download.pytorch.org/whl/cu118
        set PYTORCH_TYPE=CUDA 11.8
    ) else (
        echo Using CUDA 12.1 support
        set PYTORCH_INDEX=https://download.pytorch.org/whl/cu121
        set PYTORCH_TYPE=CUDA 12.1
    )
)
echo.

echo [4/7] Installing PyTorch (%PYTORCH_TYPE% version)...
echo This may take several minutes, please wait...
echo Download size: 2-3 GB for CUDA version
echo.
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url %PYTORCH_INDEX%
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch!
    echo.
    echo If you have a NVIDIA GPU, try installing manually:
    echo   For Python 3.13: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    echo   For Python 3.12: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo.
    echo For CPU only:
    echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo.
    pause
    exit /b 1
)
echo.

echo [5/7] Installing AI and computer vision libraries...
python -m pip install ultralytics opencv-python
if errorlevel 1 (
    echo ERROR: Failed to install AI libraries!
    pause
    exit /b 1
)
echo.

echo [6/7] Installing system libraries...
python -m pip install mss numpy pillow customtkinter pywin32 keyboard
if errorlevel 1 (
    echo ERROR: Failed to install system libraries!
    pause
    exit /b 1
)
echo.

echo [7/7] Verifying installation...
echo.

REM Test imports
python -c "import torch; import ultralytics; import cv2; import mss; import customtkinter; import keyboard; print('All core libraries imported successfully!')" 2>nul
if errorlevel 1 (
    echo ERROR: Some libraries failed to import!
    echo.
    echo Please run this script again or install manually:
    echo pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo.
echo 
echo Testing CUDA availability...
echo 
python -c "import torch; cuda = torch.cuda.is_available(); print('PyTorch Version:', torch.__version__); print('CUDA Available:', cuda); print('CUDA Version:', torch.version.cuda if cuda else 'N/A'); print('GPU Name:', torch.cuda.get_device_name(0) if cuda else 'CPU'); print('GPU Count:', torch.cuda.device_count() if cuda else 0)"
echo.

echo 
echo Installation Complete!
echo.

nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo GPU acceleration is enabled!
) else (
    echo Running in CPU mode
)

echo.
echo By: Elldries
echo.
pause
