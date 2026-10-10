@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please follow the installation steps in README.md first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m racing_lab.cli play %*
if errorlevel 1 pause
