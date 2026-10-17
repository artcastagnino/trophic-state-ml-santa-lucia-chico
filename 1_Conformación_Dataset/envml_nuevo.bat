@echo off
set "JUPYTER_RUNTIME_DIR=%USERPROFILE%\.jupyter_runtime"
if not exist "%JUPYTER_RUNTIME_DIR%" mkdir "%JUPYTER_RUNTIME_DIR%"

call "%USERPROFILE%\anaconda3\Scripts\activate.bat" envml_nuevo
python -m notebook --ServerApp.use_redirect_file=False
pause