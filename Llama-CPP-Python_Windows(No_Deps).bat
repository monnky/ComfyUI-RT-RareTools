@echo off
setlocal enabledelayedexpansion

echo =======================================================
echo Llama-CPP-Python Auto-Installer (Windows / Dynamic CUDA)
echo Target Version: v0.4.1 (No Dependencies)
echo =======================================================
echo.

:: 1. LOCATE PYTHON EXECUTABLE
set "PYTHON_EXE="

if exist "%~dp0..\..\..\python_embeded\python.exe" (
    set "PYTHON_EXE=%~dp0..\..\..\python_embeded\python.exe"
) else if exist "%~dp0..\..\..\python\python.exe" (
    set "PYTHON_EXE=%~dp0..\..\..\python\python.exe"
) else if exist "%~dp0..\..\python_embeded\python.exe" (
    set "PYTHON_EXE=%~dp0..\..\python_embeded\python.exe"
) else if exist "%~dp0..\python_embeded\python.exe" (
    set "PYTHON_EXE=%~dp0..\python_embeded\python.exe"
) else if exist "%~dp0python_embeded\python.exe" (
    set "PYTHON_EXE=%~dp0python_embeded\python.exe"
)

if defined PYTHON_EXE (
    for %%i in ("!PYTHON_EXE!") do set "PYTHON_EXE=%%~fi"
    echo [INFO] Found ComfyUI Portable Python at:
    echo "!PYTHON_EXE!"
) else (
    echo [INFO] ComfyUI Portable Python not found.
    echo [INFO] Falling back to system Python...
    set "PYTHON_EXE=python"
)
echo.

:: 2. RUN DYNAMIC PYTHON INSTALLER FOR V0.4.1
echo [INFO] Resolving optimal pre-compiled CUDA wheel for v0.4.1...
"%PYTHON_EXE%" "%~dp0install.py" --force %*

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =======================================================
    echo [SUCCESS] Llama-CPP-Python v0.4.1 Setup Complete!
    echo =======================================================
) else (
    echo.
    echo =======================================================
    echo [ERROR] Installation encountered an error (Code: %ERRORLEVEL%).
    echo =======================================================
)

pause
