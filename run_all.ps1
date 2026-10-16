# run_all.ps1
# Download the data, train every model for India and Colombia, evaluate them and build
# the manuscript's tables and figures.
# Stops at the first step that fails; each step's output is also saved to logs\run_<time>\.
#
# Run from the repo root, with the conda environment activated. -NoProfile keeps your
# PowerShell profile from re-activating another conda environment:
#     powershell -NoProfile -ExecutionPolicy Bypass -File run_all.ps1
# Resume from a later step (e.g. after fixing a failure in step 5):
#     powershell -NoProfile -ExecutionPolicy Bypass -File run_all.ps1 -From 5

param(
    [int]$From = 1
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# The scripts print non-ASCII characters (check marks, accented site names)
# that the default Windows console encoding cannot encode
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$steps = @(
    @{ Name = "download_data";          Args = "data/download_nsrdb.py" },
    @{ Name = "individual_india";       Args = "train_individual.py" },
    @{ Name = "individual_colombia";    Args = "train_individual.py --is-colombia" },
    @{ Name = "joint_india";            Args = "train_joint.py" },
    @{ Name = "joint_colombia";         Args = "train_joint.py --is-colombia" },
    @{ Name = "gnn_india";              Args = "train_gnn.py" },
    @{ Name = "gnn_colombia";           Args = "train_gnn.py --is-colombia" },
    @{ Name = "evaluate_india";         Args = "evaluate.py" },
    @{ Name = "evaluate_colombia";      Args = "evaluate.py --is-colombia" },
    @{ Name = "significance_india";     Args = "significance_tests.py" },
    @{ Name = "significance_colombia";  Args = "significance_tests.py --is-colombia" },
    @{ Name = "tables";                 Args = "make_results_tables.py" },
    @{ Name = "figures";                Args = "plot_results.py" },
    @{ Name = "cloud_climate_india";    Args = "compute_cloud_climate_stats.py" },
    @{ Name = "cloud_climate_colombia"; Args = "compute_cloud_climate_stats.py --is-colombia" },
    @{ Name = "cloud_type_india";       Args = "plot_cloud_type_heatmap.py" },
    @{ Name = "cloud_type_colombia";    Args = "plot_cloud_type_heatmap.py --is-colombia" }
)

$logDir = Join-Path $PSScriptRoot ("logs\run_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Write-Host "Python: $((Get-Command python).Source)"
Write-Host "Logs:   $logDir`n"

# Fail fast if this isn't the environment with the ML dependencies
cmd /c "python -c ""import tensorflow, spektral, h5py"" 2>&1" | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "This Python can't import tensorflow, spektral and h5py. Activate the environment" -ForegroundColor Red
    Write-Host "from the README (conda activate ghimultisite) and run with -NoProfile:" -ForegroundColor Red
    Write-Host "    powershell -NoProfile -ExecutionPolicy Bypass -File run_all.ps1" -ForegroundColor Red
    exit 1
}

$runStart = Get-Date
for ($n = $From; $n -le $steps.Count; $n++) {
    $step = $steps[$n - 1]
    $log = Join-Path $logDir ("{0:D2}_{1}.log" -f $n, $step.Name)
    Write-Host "[$n/$($steps.Count)] python $($step.Args)" -ForegroundColor Cyan
    $start = Get-Date

    # cmd merges stderr into stdout before PowerShell sees it, so Python and
    # TensorFlow warnings on stderr aren't treated as PowerShell errors, and
    # $LASTEXITCODE is Python's exit code
    cmd /c "python -u $($step.Args) 2>&1" | Tee-Object -FilePath $log
    $code = $LASTEXITCODE

    $minutes = ((Get-Date) - $start).TotalMinutes
    if ($code -ne 0) {
        Write-Host ("`nFAILED at step {0} ({1}) with exit code {2} after {3:N1} min." -f $n, $step.Name, $code, $minutes) -ForegroundColor Red
        Write-Host "Log: $log" -ForegroundColor Red
        Write-Host "After fixing it, resume with: powershell -NoProfile -ExecutionPolicy Bypass -File run_all.ps1 -From $n" -ForegroundColor Red
        exit $code
    }
    Write-Host ("  done in {0:N1} min`n" -f $minutes) -ForegroundColor Green
}

Write-Host ("All steps finished in {0:N1} h." -f ((Get-Date) - $runStart).TotalHours) -ForegroundColor Green
