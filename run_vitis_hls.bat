@echo off
setlocal

set VITIS_ROOT=E:\2025.2\Vitis
set HLS_PART=xc7z020clg400-1
set HLS_CLOCK_PERIOD=10.0
if not "%~1"=="" set HLS_FLOW=%~1
if "%HLS_FLOW%"=="" set HLS_FLOW=all

set XILINX_VITIS=%VITIS_ROOT%
set PYTHONPATH=%VITIS_ROOT%\cli;%VITIS_ROOT%\cli\python-packages\win64;%VITIS_ROOT%\cli\python-packages\site-packages;%VITIS_ROOT%\cli\proto;%VITIS_ROOT%\scripts\python_pkg
set PATH=%VITIS_ROOT%\bin;%PATH%

python "%~dp0run_hls_cli.py"

endlocal
