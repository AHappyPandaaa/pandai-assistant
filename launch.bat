@echo off
cd /d "%~dp0"

:: Detect Python
py -3.11 --version >nul 2>&1
if not errorlevel 1 ( py -3.11 main.py & exit /b )

py --version >nul 2>&1
if not errorlevel 1 ( py main.py & exit /b )

python main.py
