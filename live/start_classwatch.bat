@echo off
cd /d "%~dp0.."
python live\classwatch_live.py %*
pause
