@echo off
cd /d "%~dp0"

:: Detect Python (use pythonw to suppress the console window)
py -3.11 --version >nul 2>&1
if not errorlevel 1 ( start "" pythonw -3.11 main.py & exit /b )

py --version >nul 2>&1
if not errorlevel 1 ( start "" pythonw main.py & exit /b )

start "" pythonw main.py
