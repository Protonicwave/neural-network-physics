@echo off
rem Serve the generated run reports over HTTP. Reports are self contained HTML
rem under runs\<run-id>\report.html, so a static file server is all that is needed.
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

rem Landing page listing only the runs that actually have a rendered report.
uv run python -c "import pathlib, html; root = pathlib.Path('runs'); reports = sorted(p for p in root.glob('*/report.html')); rows = ''.join('<li><a href=\"{0}/report.html\">{0}</a></li>'.format(html.escape(p.parent.name)) for p in reports); (root / 'index.html').write_text('<!doctype html><meta charset=\"utf-8\"><title>nnphysics reports</title><style>body{font-family:system-ui,sans-serif;margin:3rem auto;max-width:48rem;line-height:1.6}</style><h1>nnphysics reports</h1><ul>' + (rows or '<li>No rendered reports. Run: uv run nnp report render --run &lt;run-id&gt;</li>') + '</ul>', encoding='utf-8'); print('{} report(s) listed'.format(len(reports)))"
if errorlevel 1 exit /b 1

echo Serving runs\ on http://localhost:%PORT%/  (Ctrl+C to stop)
start "" "http://localhost:%PORT%/"
uv run python -m http.server %PORT% --directory runs --bind 127.0.0.1

endlocal
