@echo off
call "%~dp0..\run_smoke.cmd" %*
exit /b %errorlevel%
