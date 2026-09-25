@echo off
rem Lets ninfer-switch run from cmd, PowerShell or the Run dialog once this folder is on PATH.
where pwsh >nul 2>nul
if %errorlevel%==0 (
  pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0ninfer-switch.ps1" %*
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0ninfer-switch.ps1" %*
)
exit /b %errorlevel%
