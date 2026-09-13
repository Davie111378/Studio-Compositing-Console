@echo off
chcp 65001 >nul
title Studio Agent Launcher
REM ============================================================
REM  Studio Agent one-click launcher
REM  - Double-click this file to (re)start the agent service
REM    and open the /agent frontend in your browser.
REM  - Safe to run again after any code update: it clears the
REM    old service on port 8765 first, then starts fresh code.
REM  - This launcher now lives in app/; it switches to project root.
REM ============================================================

cd /d "%~dp0\.."

set "PY=C:\Users\zhaod\venvs\prod-gpu\Scripts\python.exe"
set "PORT=8765"
set "URL=http://127.0.0.1:%PORT%/agent"

if not exist "%PY%" (
    echo [ERROR] python not found: %PY%
    pause
    exit /b 1
)
if not exist "app\studio_web.py" (
    echo [ERROR] app\studio_web.py not found in %CD%
    pause
    exit /b 1
)

echo [1/3] Stopping old service on port %PORT% (if any)...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -Confirm:$false }"
ping -n 3 127.0.0.1 >nul

echo [2/3] Starting app\studio_web.py (hidden background process)...
powershell -NoProfile -Command "Start-Process -WindowStyle Hidden -FilePath '%PY%' -ArgumentList 'app\studio_web.py','%PORT%' -WorkingDirectory (Get-Location).Path"

echo [3/3] Waiting for service ready, then opening browser...
powershell -NoProfile -Command "$ok=$false; for($i=0; $i -lt 40; $i++){ Start-Sleep -Milliseconds 500; try { $wc = New-Object System.Net.WebClient; $wc.Proxy = $null; $s = $wc.DownloadString('http://127.0.0.1:%PORT%/api/agent/providers'); if($s){ $ok=$true; break } } catch {} }; if($ok){ Start-Process '%URL%'; Write-Host 'READY: %URL%'; exit 0 } else { Write-Host 'NOT READY after 20s'; exit 1 }"
if errorlevel 1 (
    echo.
    echo [FALLBACK] Background start failed - running server in this window.
    echo Close this window to stop the service. Browser: %URL%
    start "" "%URL%"
    "%PY%" app\studio_web.py %PORT%
    echo Service exited. Press any key to close.
    pause >nul
    exit /b 0
)

echo.
echo Service is running in background. Close this window anytime.
pause
