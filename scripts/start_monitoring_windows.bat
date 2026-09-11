@echo off
REM Launch Prometheus and Grafana on Windows without Docker
powershell -ExecutionPolicy Bypass -File "%~dp0start_monitoring_windows.ps1"
pause
