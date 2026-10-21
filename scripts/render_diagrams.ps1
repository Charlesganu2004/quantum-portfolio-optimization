# Render every Mermaid diagram to a high-resolution PNG for the report.
# The PNGs are not committed. docs\DIAGRAMS.md holds the same diagrams as fenced
# mermaid blocks, which GitHub renders.
#
#   cd <repository root>
#   npm install @mermaid-js/mermaid-cli
#   powershell -ExecutionPolicy Bypass -File scripts\render_diagrams.ps1
#
# Sources are docs\diagrams\*.mmd. Output lands in reports\figures\ beside the
# matplotlib figures.

$ErrorActionPreference = "Continue"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
$src = Join-Path $root "docs\diagrams"
$out = Join-Path $root "reports\figures"
New-Item -ItemType Directory -Force -Path $out | Out-Null

# scale 3 gives roughly 300 dpi at print size, which stays sharp in the PDF.
$common = @("-b", "white", "-s", "3")

$jobs = @(
    @{ file = "architecture.mmd";     width = 1600 },
    @{ file = "pipeline.mmd";         width = 1700 },
    @{ file = "reviewer_states.mmd";  width = 1100 },
    @{ file = "qubo.mmd";             width = 1300 },
    @{ file = "ablation.mmd";         width = 1400 },
    @{ file = "halt_latch.mmd";        width = 1600 }
)

foreach ($job in $jobs) {
    $inFile = Join-Path $src $job.file
    $name = [System.IO.Path]::GetFileNameWithoutExtension($job.file)
    $outFile = Join-Path $out "diagram_$name.png"
    Write-Host "rendering $($job.file) -> $outFile"
    npx mmdc -i $inFile -o $outFile -w $job.width @common
    if ($LASTEXITCODE -ne 0) { Write-Host "  FAILED: $($job.file)" -ForegroundColor Red }
}

Write-Host ""
Get-ChildItem $out -Filter "diagram_*.png" | Format-Table Name, Length -AutoSize
