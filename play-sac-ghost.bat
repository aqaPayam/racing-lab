@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please follow the installation steps in README.md first.
  pause
  exit /b 1
)
set "ghost_model=results\sac-monza-200k-seed0-20261011\final.pt"
if not "%~1"=="" set "ghost_model=%~1"
if not exist "%ghost_model%" (
  echo SAC checkpoint not found: "%ghost_model%"
  echo Supply your trained checkpoint: play-sac-ghost.bat "path\to\final.pt"
  echo Checkpoints are local training artifacts and are not included in Git.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m racing_lab.cli play --track monza --sac-ghost "%ghost_model%"
if errorlevel 1 pause
