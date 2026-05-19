@echo off
setlocal
cd /d "%~dp0"

if exist "set_secrets.bat" (
    call "set_secrets.bat"
)

if "%GEMINI_API_KEY%"=="" (
    echo [ERRO] GEMINI_API_KEY nao esta definida.
    echo Defina antes de rodar. Exemplo:
    echo   set GEMINI_API_KEY=sua_chave
    echo   run.bat
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
    exit /b %errorlevel%
)

py -3 -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
if %errorlevel% neq 0 (
    python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
)

endlocal
