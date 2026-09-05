@echo off
setlocal
cd /d "%~dp0\..\.."
python -m pytest tests\test_e3_routing_smoke.py tests\test_routing_interpreter.py -q
