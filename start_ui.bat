@echo off
title AIML Work Agent Dashboard
echo ===================================================
echo   AIML Work Agent - Executive Dashboard
echo   Reinforcement Learning Autonomous Workflow Controller
echo ===================================================
echo.
echo Opening dashboard in your default browser...
start http://127.0.0.1:8000
echo Starting Python Web Server on port 8000...
python app.py 8000
pause

