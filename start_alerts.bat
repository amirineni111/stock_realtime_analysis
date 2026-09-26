@echo off
cd /d "%~dp0"
echo Starting real-time stock alerts (scans after every 5-minute bar, 09:30-16:00 ET)
echo New signals are pushed to the ntfy topic in .env. Close this window to stop.
venv\Scripts\python scripts\run_alerts.py
pause
