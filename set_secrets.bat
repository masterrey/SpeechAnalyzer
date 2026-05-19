@echo off
setlocal
cd /d "%~dp0"

if exist ".secrets.bat" (
    call ".secrets.bat"
) else (
    echo [AVISO] Arquivo .secrets.bat nao encontrado. Crie com sua chave local.
)

if "%GEMINI_API_KEY%"=="" (
    echo [AVISO] GEMINI_API_KEY ainda nao foi definida.
) else (
    echo [OK] GEMINI_API_KEY carregada.
)

endlocal & set "GEMINI_API_KEY=%GEMINI_API_KEY%"
