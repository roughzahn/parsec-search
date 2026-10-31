@echo off
pushd "%~dp0"
docker compose up -d --build
if errorlevel 1 (
  echo Failed to start parsec-search.
  popd
  exit /b 1
)
echo.
echo parsec-search is starting. The CLAP model may need about 60 seconds to load.
echo Browser: http://127.0.0.1:8765
popd
