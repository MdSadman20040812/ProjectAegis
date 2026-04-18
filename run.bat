@echo off
title AEGIS Cognitive Tutor
echo.
echo  =========================================
echo   AEGIS - Cognitive Tutor System
echo   Starting Streamlit Interface...
echo  =========================================
echo.
call "%~dp0venv\Scripts\activate.bat"
streamlit run "%~dp0app.py"
pause
