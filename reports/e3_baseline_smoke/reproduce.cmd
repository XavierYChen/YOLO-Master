@echo off
setlocal

if not exist reports\e3_baseline_smoke\results mkdir reports\e3_baseline_smoke\results

yolo detect train model=ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml data=coco8.yaml epochs=1 imgsz=320 batch=2 device=0 workers=0 project=runs/e3_baseline name=detect_smoke_reproduce exist_ok=True > reports\e3_baseline_smoke\results\smoke_reproduce.log 2>&1

set "E3_SMOKE_EXIT=%ERRORLEVEL%"
type reports\e3_baseline_smoke\results\smoke_reproduce.log
exit /b %E3_SMOKE_EXIT%
