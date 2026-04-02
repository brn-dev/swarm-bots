param(
    [int]$Runs = 5,
    [string]$PythonExecutable = "python",
    [int]$DelaySeconds = 0,
    [string[]]$ScriptArgs = @()
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$trainingScript = Join-Path $PSScriptRoot "run_mat_nop_wall.py"
$failedRuns = @()
$originalPythonPath = $env:PYTHONPATH

for ($runIndex = 1; $runIndex -le $Runs; $runIndex++) {
    $startedAt = Get-Date
    Write-Host ("[{0}/{1}] Starting run_mat_nop_wall.py at {2}" -f $runIndex, $Runs, $startedAt.ToString("s"))

    if ([string]::IsNullOrWhiteSpace($originalPythonPath)) {
        $env:PYTHONPATH = $repoRoot
    }
    else {
        $env:PYTHONPATH = "$repoRoot$([IO.Path]::PathSeparator)$originalPythonPath"
    }

    Push-Location $PSScriptRoot
    try {
        & $PythonExecutable $trainingScript @ScriptArgs
        $exitCode = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } else { 0 }
    }
    finally {
        Pop-Location
        $env:PYTHONPATH = $originalPythonPath
    }

    $finishedAt = Get-Date
    if ($exitCode -eq 0) {
        Write-Host ("[{0}/{1}] Finished successfully at {2}" -f $runIndex, $Runs, $finishedAt.ToString("s"))
    }
    else {
        $failedRuns += [pscustomobject]@{
            Run = $runIndex
            ExitCode = $exitCode
            StartedAt = $startedAt
            FinishedAt = $finishedAt
        }
        Write-Warning ("[{0}/{1}] Failed with exit code {2} at {3}" -f $runIndex, $Runs, $exitCode, $finishedAt.ToString("s"))
    }

    if ($runIndex -lt $Runs -and $DelaySeconds -gt 0) {
        Start-Sleep -Seconds $DelaySeconds
    }
}

if ($failedRuns.Count -gt 0) {
    Write-Host ""
    Write-Host "Failed runs:"
    $failedRuns | Format-Table -AutoSize | Out-Host
    exit 1
}

Write-Host ""
Write-Host ("All {0} runs finished successfully." -f $Runs)
exit 0
