@echo off
setlocal

set "JUPYTER_RUNTIME_DIR=%USERPROFILE%\.jupyter_runtime"
if not exist "%JUPYTER_RUNTIME_DIR%" mkdir "%JUPYTER_RUNTIME_DIR%"

REM Limpiar archivos viejos de servidores Jupyter
del /Q "%JUPYTER_RUNTIME_DIR%\jpserver-*.json" 2>nul
del /Q "%JUPYTER_RUNTIME_DIR%\nbserver-*.json" 2>nul

REM Activar entorno
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" envml_nuevo

REM Variables necesarias para PyTorch/OpenMP en Windows
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "PATH=%CONDA_PREFIX%\Lib\site-packages\torch\lib;%PATH%"

REM Verificación rápida antes de abrir Jupyter
python -c "import sys, os; print('Python:', sys.executable); print('KMP:', os.environ.get('KMP_DUPLICATE_LIB_OK')); import torch; print('Torch OK:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

REM Abrir Jupyter
python -m notebook --ServerApp.use_redirect_file=False

pause
endlocal