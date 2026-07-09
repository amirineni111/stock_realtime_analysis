@echo off
cd /d "%~dp0"
echo Starting Stock Screening Dashboard...
echo Open http://localhost:8502 in your browser
call venv\Scripts\activate.bat
venv\Scripts\streamlit run app.py --server.port 8502
pause
