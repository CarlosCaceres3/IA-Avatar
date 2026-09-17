@echo off
setlocal
title Avatar IA - Festech
cd /d "%~dp0"

echo.
echo   ========================================
echo    AVATAR IA - arrancando...
echo   ========================================
echo.

REM Se prefiere el lanzador "py" porque en Windows el comando "python"
REM puede apuntar al atajo de la Microsoft Store, que no ejecuta nada.
set PY=py -3
%PY% -c "import sys" >nul 2>&1
if errorlevel 1 set PY=python
%PY% -c "import sys" >nul 2>&1
if errorlevel 1 goto sin_python

REM Las librerias tambien se comprueban: es el fallo mas comun al mover
REM la carpeta a otro equipo.
%PY% -c "import cv2, mediapipe" >nul 2>&1
if errorlevel 1 goto sin_librerias

REM El modelo no viene en la carpeta del repositorio: se baja una sola vez.
if not exist "models\pose_landmarker_full.task" (
  echo   Falta el modelo de pose. Descargando ^(9 MB^)...
  %PY% tools\descargar_modelo.py
  echo.
)

REM ----------------------------------------------------------------
REM  OPCIONES DEL STAND: edita la linea de abajo si hace falta.
REM
REM    --avatar N        con cual arranca (0 a 10)
REM    --personas N      cuantas personas a la vez (1 a 4)
REM    --deteccion 0.4   si la segunda persona no aparece
REM    --calidad 0.5     mas rapido si el equipo va justo
REM    --exposure -5     sube el FPS, pero SOLO con mucha luz
REM
REM  Teclas en vivo: A/D avatar, F fondo, C cara real, Q salir.
REM ----------------------------------------------------------------
%PY% avatar_cam.py --fullscreen --personas 2

if errorlevel 1 goto con_error
exit /b 0

:sin_python
echo.
echo   No encuentro Python en este equipo.
echo   Instalalo desde python.org (marca "Add to PATH") y vuelve a abrir esto.
echo.
pause
exit /b 1

:sin_librerias
echo.
echo   Faltan las librerias. Instalandolas ahora ^(puede tardar unos minutos^)...
echo.
%PY% -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo   No se pudieron instalar. Revisa la conexion a internet.
  echo.
  pause
  exit /b 1
)
echo.
echo   Listo. Vuelve a abrir INICIAR.bat
echo.
pause
exit /b 0

:con_error
echo.
echo   La aplicacion termino con un error. El detalle esta arriba.
echo.
echo   Cosas que suelen fallar:
echo     - La camara esta ocupada por Teams, Zoom o el navegador: cierralos.
echo     - Otra camara: prueba a anadir  --camera 1  a la linea de arriba.
echo.
pause
exit /b 1
