@echo off
setlocal
cd /d "%~dp0\..\.."
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
if not defined E3_PYTHON set "E3_PYTHON=python"
"%E3_PYTHON%" scripts\e3_routing_smoke.py --config reports\e3_routing_smoke\configs\smoke.yaml %*
exit /b %errorlevel%
