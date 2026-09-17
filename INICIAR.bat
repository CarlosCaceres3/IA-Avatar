@echo off
REM Lanzador para el stand: pantalla completa desde el arranque.
REM Si el stand tiene buena luz, agrega  --exposure -5  para subir a 30 FPS.
cd /d "%~dp0"
python avatar_cam.py --fullscreen
pause
