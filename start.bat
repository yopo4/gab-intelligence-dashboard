@echo off
title GAB Intelligence Dashboard
cd /d "%~dp0"

:: Ajouter Node.js au PATH
set "PATH=%PATH%;C:\Program Files\nodejs;%APPDATA%\npm"

echo.
echo  ============================================
echo   GAB Intelligence Dashboard - Lancement
echo  ============================================
echo.

:: Verifier Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python introuvable. Ajoutez Python a votre PATH.
    pause & exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo   Python  : %%i

:: Verifier Node / npm
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Node.js introuvable.
    echo   Telechargez Node.js sur https://nodejs.org
    pause & exit /b 1
)
for /f "tokens=*" %%i in ('node --version 2^>^&1') do echo   Node.js : %%i

echo.

:: Stocker les chemins
set "BACK=%~dp0backend"
set "FRONT=%~dp0frontend"

:: Installer les deps Python si besoin
echo [0/2] Dependances Python...
pip install -q -r "%~dp0backend\requirements.txt"
if errorlevel 1 (
    echo [AVERTISSEMENT] pip install a echoue - verifiez requirements.txt
)

:: Lancer le backend dans une nouvelle fenetre
echo [1/2] Backend FastAPI sur http://localhost:8000 ...
start "GAB Backend" cmd /k "cd /d "%BACK%" && python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000"

:: Attendre que le backend soit pret
timeout /t 4 /nobreak >nul

:: Lancer le frontend dans une nouvelle fenetre
echo [2/2] Frontend React sur http://localhost:5173 ...
start "GAB Frontend" cmd /k "set PATH=%PATH% && cd /d "%FRONT%" && npm install && npm run dev"

echo.
echo  ============================================
echo   Liens utiles
echo  ============================================
echo   Frontend : http://localhost:5173
echo   Backend  : http://localhost:8000
echo   API docs : http://localhost:8000/docs
echo   Health   : http://localhost:8000/api/health
echo  ============================================
echo.
echo  Fermez les fenetres "GAB Backend" et "GAB Frontend"
echo  pour arreter les serveurs.
echo.
pause
