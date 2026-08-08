@echo off
rem Build the landing page and serve the runs directory over HTTP. The page references
rem the plots beside it rather than embedding them, so it is served from the directory
rem that holds them and a static file server is all that is needed.
rem Usage: serve.bat [port]

setlocal
cd /d "%~dp0"

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"

if not exist "runs" (
  echo No runs directory. Generate one first, for example:
  echo   uv run nnp data generate --config configs\nbody.yaml
  echo   uv run nnp train         --config configs\nbody.yaml
  echo   uv run nnp report render --run ^<run-id^>
  exit /b 1
)

rem The landing page, derived from the run records rather than written here.
uv run nnp report page --root runs
if errorlevel 1 exit /b 1

echo Serving runs\ on http://localhost:%PORT%/  (Ctrl+C to stop)
start "" "http://localhost:%PORT%/"
uv run python -m http.server %PORT% --directory runs --bind 127.0.0.1

endlocal
