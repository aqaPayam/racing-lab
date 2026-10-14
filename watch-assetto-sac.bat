@echo off
setlocal
cd /d "%~dp0"
if not exist ".tools\ac-python39\python.exe" (
  echo The separate Assetto Corsa runtime has not been prepared.
  echo See integrations\assetto_corsa\README.md for setup.
  pause
  exit /b 1
)
echo Start a Monza Hotlap with the BMW Z4 GT3 and the saved Vjoy controls.
echo Use the research settings described in integrations\assetto_corsa\README.md.
echo Click Drive in the game. Then return here and press a key to watch SAC.
echo This runs the released policy only. It does not train a model.
pause
".tools\ac-python39\python.exe" -I integrations\assetto_corsa\runner.py evaluate --seconds 600
echo Select keyboard/mouse controls in the launcher when returning to manual driving.
pause
