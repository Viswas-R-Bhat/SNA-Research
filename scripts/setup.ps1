param([string]$Python = "py", [switch]$Cpu)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (!(Test-Path .venv/Scripts/python.exe)) {
    if ($Python -eq "py") { & py -3.12 -m venv .venv }
    else { & $Python -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw "Install Python 3.12 and retry." }
}
$TaskPython = Join-Path $PWD ".venv/Scripts/python.exe"
& $TaskPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
$TaskIndex = if ($Cpu) { "https://download.pytorch.org/whl/cpu" } else { "https://download.pytorch.org/whl/cu121" }
& $TaskPython -m pip install torch==2.5.1 --index-url $TaskIndex
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed" }
& $TaskPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
& $TaskPython -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw "Project installation failed" }
& $TaskPython -m sebot_mls.cli doctor
