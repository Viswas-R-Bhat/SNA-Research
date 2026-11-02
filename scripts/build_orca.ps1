$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force tools | Out-Null
& g++ -O3 -std=c++11 -static vendor/orca/orca.cpp -o tools/orca.exe
if ($LASTEXITCODE -ne 0) { throw "ORCA compilation failed. Install MinGW g++ or compile with MSVC as described in README." }
Write-Host "Built exact graphlet counter tools/orca.exe"
