@echo off
setlocal enabledelayedexpansion

set ENV_NAME=FaceID

rem --- Localizar conda aunque no este en el PATH ---
set CONDA_CMD=conda
where conda >nul 2>nul
if errorlevel 1 (
    for %%P in (
        "%USERPROFILE%\anaconda3\condabin\conda.bat"
        "%USERPROFILE%\miniconda3\condabin\conda.bat"
        "C:\ProgramData\anaconda3\condabin\conda.bat"
        "C:\ProgramData\miniconda3\condabin\conda.bat"
    ) do (
        if exist %%P (
            set CONDA_CMD=%%~P
            goto :found_conda
        )
    )
    echo No se encontro conda ni en el PATH ni en las rutas comunes de instalacion.
    echo Abre "Anaconda Prompt" y corre este script desde ahi, o instala Anaconda/Miniconda.
    goto :error
)
:found_conda
echo Usando conda: %CONDA_CMD%

echo === Creando entorno conda "%ENV_NAME%" ===
call "%CONDA_CMD%" create -n %ENV_NAME% python=3.10 -y
if errorlevel 1 goto :error

echo === Instalando dependencias del backend ===
call "%CONDA_CMD%" run -n %ENV_NAME% pip install -r "%~dp0backend\requirements.txt"
if errorlevel 1 goto :error

echo === Descargando modelos (GhostFaceNets + detector de rostros) ===
call "%CONDA_CMD%" run -n %ENV_NAME% python "%~dp0backend\download_models.py"
if errorlevel 1 goto :error

echo.
echo Listo. Para levantar el backend:
echo   conda activate %ENV_NAME%
echo   cd backend
echo   uvicorn app:app --reload --port 8000
goto :eof

:error
echo.
echo Algo fallo durante la instalacion. Revisa el mensaje de error de arriba.
exit /b 1
