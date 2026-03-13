@echo off
title PandAI Assistant - First Time Setup
echo ============================================
echo  PandAI Assistant - First Time Setup
echo  This will take 5-10 minutes
echo ============================================
echo.

:: Check for Python 3.11
py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    echo Python 3.11 found.
    goto :install_deps
)

:: Not found - download and install automatically
echo Python 3.11 not found. Downloading now (~25 MB)...
echo.
powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile '%TEMP%\python3.11.exe' -UseBasicParsing"
if errorlevel 1 (
    echo ERROR: Could not download Python. Check your internet connection.
    pause
    exit /b 1
)

echo Installing Python 3.11...
:: InstallAllUsers=0 = current user only, no admin needed
:: PrependPath=1 = adds to PATH automatically
:: Include_launcher=1 = installs the py launcher so py -3.11 works
"%TEMP%\python3.11.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0
if errorlevel 1 (
    echo ERROR: Python installation failed.
    echo Please install Python 3.11 manually from:
    echo https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
    pause
    exit /b 1
)

:: Refresh PATH so py launcher is found in this session
set "PATH=%LOCALAPPDATA%\Programs\Python\Python311\;%LOCALAPPDATA%\Programs\Python\Python311\Scripts\;%PATH%"
set "PATH=%APPDATA%\Python\Python311\Scripts\;%PATH%"

echo Python 3.11 installed successfully.
echo.

:install_deps
set PY=py -3.11

echo [1/4] Installing core dependencies...
%PY% -m pip install -q --upgrade pip
%PY% -m pip install -q PyQt6 anthropic sounddevice numpy scipy faster-whisper keyboard
if errorlevel 1 (
    echo ERROR: Failed to install core dependencies.
    pause
    exit /b 1
)

echo [2/4] Installing CUDA libraries for GPU acceleration...
%PY% -m pip install -q nvidia-cublas-cu12 nvidia-cudnn-cu12==8.9.7.29
if errorlevel 1 (
    echo WARNING: CUDA libraries failed - app will use CPU mode (slower).
    echo.
)

echo [3/4] Pre-downloading Whisper AI model (1.5 GB - please wait)...
%PY% -c "from faster_whisper import WhisperModel; print('Downloading...'); m = WhisperModel('medium', device='cpu', compute_type='int8'); print('Model ready.')"
if errorlevel 1 (
    echo WARNING: Model download failed - will retry on first launch.
    echo.
)

echo [4/4] Creating desktop shortcut...
set SCRIPT_DIR=%~dp0

:: Find pythonw.exe path (suppresses console window on launch)
for /f "tokens=*" %%i in ('%PY% -c "import sys,os; print(os.path.join(os.path.dirname(sys.executable),\"pythonw.exe\"))"') do set PYTHONW=%%i

:: Create a proper .lnk shortcut using PowerShell
powershell -NoProfile -Command ^
    "$ws = New-Object -ComObject WScript.Shell;" ^
    "$sc = $ws.CreateShortcut('%USERPROFILE%\Desktop\PandAI Assistant.lnk');" ^
    "$sc.TargetPath = '%PYTHONW%';" ^
    "$sc.Arguments = '\"%SCRIPT_DIR%main.py\"';" ^
    "$sc.WorkingDirectory = '%SCRIPT_DIR%';" ^
    "$sc.IconLocation = '%PYTHONW%,0';" ^
    "$sc.Description = 'PandAI Assistant - Real-time AI conversation helper';" ^
    "$sc.Save()"

if errorlevel 1 (
    echo WARNING: Could not create .lnk shortcut - falling back to .bat shortcut.
    echo @echo off > "%USERPROFILE%\Desktop\PandAI Assistant.bat"
    echo cd /d "%SCRIPT_DIR%" >> "%USERPROFILE%\Desktop\PandAI Assistant.bat"
    echo py -3.11 "%SCRIPT_DIR%main.py" >> "%USERPROFILE%\Desktop\PandAI Assistant.bat"
)

echo.
echo ============================================
echo  Setup complete!
echo.
echo  A "PandAI Assistant" shortcut has been
echo  placed on your Desktop.
echo.
echo  IMPORTANT - For system audio capture:
echo  VB-Cable (free) must be installed.
echo  Download: https://vb-audio.com/Cable/
echo  See HOW_TO_INSTALL.txt for setup steps.
echo ============================================
echo.
pause
