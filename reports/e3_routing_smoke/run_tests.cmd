@echo off
setlocal
cd /d "%~dp0\..\.."
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "YOLO_CONFIG_DIR=%cd%\reports\e3_routing_smoke\env\runtime"
if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"
if not defined E3_PYTHON set "E3_PYTHON=python"
"%E3_PYTHON%" -m pytest tests\test_e3_routing_smoke.py -q -o addopts=
exit /b %errorlevel%
