@echo off
cd /d "%~dp0"

:: Find pythonw.exe via the py launcher (no console window, no relaunch needed)
for /f "usebackq delims=" %%P in (`py -3.11 -c "import sys,os; print(os.path.join(os.path.dirname(sys.executable),'pythonw.exe'))" 2^>nul`) do set PYTHONW=%%P
if defined PYTHONW if exist "%PYTHONW%" (
    start "" "%PYTHONW%" "%~dp0main.py"
    exit /b 0
)

:: Fallback: try pythonw on PATH
where pythonw >nul 2>&1
if not errorlevel 1 (
    start "" pythonw "%~dp0main.py"
    exit /b 0
)

:: Last resort: regular python (shows console briefly)
py -3.11 "%~dp0main.py" 2>"%~dp0launch_error.txt"
if errorlevel 1 (
    echo Launch failed. See launch_error.txt for details.
    pause
)
