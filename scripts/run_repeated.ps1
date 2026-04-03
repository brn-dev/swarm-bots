param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ScriptPath,
    [Parameter(Position = 1)]
    [int]$Runs = 5,
    [string]$PythonExecutable = "python",
    [int]$DelaySeconds = 0,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-TrainingScriptPath {
    [OutputType([string])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$RequestedPath,
        [Parameter(Mandatory = $true)]
        [string[]]$BaseDirectories
    )

    if ([System.IO.Path]::IsPathRooted($RequestedPath)) {
        if (-not (Test-Path -LiteralPath $RequestedPath -PathType Leaf)) {
            throw "Training script not found: $RequestedPath"
        }

        return (Resolve-Path -LiteralPath $RequestedPath).Path
    }

    foreach ($baseDirectory in $BaseDirectories) {
        $candidatePath = Join-Path $baseDirectory $RequestedPath
        if (Test-Path -LiteralPath $candidatePath -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidatePath).Path
        }
    }

    throw "Training script not found: $RequestedPath"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$trainingScript = Resolve-TrainingScriptPath -RequestedPath $ScriptPath -BaseDirectories @(
    (Get-Location).Path,
    $repoRoot,
    $PSScriptRoot
)
$failedRuns = @()
$originalPythonPath = $env:PYTHONPATH

for ($runIndex = 1; $runIndex -le $Runs; $runIndex++) {
    $startedAt = Get-Date
    Write-Host ("[{0}/{1}] Starting {2} at {3}" -f $runIndex, $Runs, $trainingScript, $startedAt.ToString("s"))

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
