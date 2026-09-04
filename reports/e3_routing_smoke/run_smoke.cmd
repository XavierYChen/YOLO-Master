@echo off
setlocal
cd /d "%~dp0\..\.."

if not exist "reports\e3_routing_smoke\results" mkdir "reports\e3_routing_smoke\results"

echo [1/3] Installing the local YOLO-Master package...
python -m pip install -e .
if errorlevel 1 exit /b %errorlevel%

echo [2/3] Recording the full environment check...
(python --version & yolo version & yolo checks) > "reports\e3_routing_smoke\results\preflight.log" 2>&1
type "reports\e3_routing_smoke\results\preflight.log"

echo [3/3] Running MoE / MoT / Latent admission smoke and overhead benchmark...
python scripts\e3_routing_smoke.py --device auto --imgsz 320 --warmup 2 --iterations 10 > "reports\e3_routing_smoke\results\full.log" 2>&1
set "E3_EXIT=%errorlevel%"
copy /y "reports\e3_routing_smoke\results\full.log" "reports\e3_routing_smoke\results\smoke.log" >nul
type "reports\e3_routing_smoke\results\full.log"

if not "%E3_EXIT%"=="0" (
  echo E3 smoke FAILED. Read reports\e3_routing_smoke\results\full.log
  exit /b %E3_EXIT%
)

echo E3 smoke PASSED. Open reports\e3_routing_smoke\results\routing_snapshot.png
exit /b 0
