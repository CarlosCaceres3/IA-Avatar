@echo off
setlocal
title Avatar IA - chequeo previo al evento
cd /d "%~dp0"

echo.
echo   ========================================
echo    CHEQUEO PREVIO - correlo en el stand
echo    con la luz y la camara que vas a usar
echo   ========================================
echo.

set PY=py -3
%PY% -c "import sys" >nul 2>&1
if errorlevel 1 set PY=python

%PY% tools\diagnostico.py

echo.
echo   ----------------------------------------
echo    Que mirar:
echo      - FPS: por debajo de 15 revisa la luz.
echo      - "cuadros con persona detectada":
echo        si es 0 con alguien delante, falta luz.
echo      - Camara virtual: debe decir OK.
echo   ----------------------------------------
echo.
pause
