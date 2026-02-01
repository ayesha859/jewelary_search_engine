@echo off
cd /d "%~dp0"
echo ==========================================
echo      JEWELRY APP LAUNCHER
echo ==========================================
echo.
echo Running Setup (This may take a minute)...
python setup.py
echo.
echo ------------------------------------------
echo Launching Website...
echo ------------------------------------------
python -m streamlit run app.py
echo.
echo ==========================================
echo IF THE APP CRASHED, READ THE ERROR ABOVE
echo ==========================================
pause