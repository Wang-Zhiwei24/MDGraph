$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Install the project in .venv first; see README.md.'
}
$previousDllDir = $env:MDGRAPH_DLL_DIR
$portableDir = Join-Path $projectRoot '.tools\msmpi\portable'
try {
    if (Test-Path -LiteralPath (Join-Path $portableDir 'msmpi.dll')) {
        $env:MDGRAPH_DLL_DIR = $portableDir
    }
    Push-Location $projectRoot
    try {
        & $pythonPath -m mdgraph.cli run examples/argon_local.yaml --watch
        if ($LASTEXITCODE -ne 0) { throw "MD workflow exited with code $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
} finally {
    $env:MDGRAPH_DLL_DIR = $previousDllDir
}
