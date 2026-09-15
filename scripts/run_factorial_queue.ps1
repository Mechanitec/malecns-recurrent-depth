$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
$checkpoint = 'results/training/v1_depth16_rate_large/readout_checkpoint.npz'
$annotations = 'data/body-annotations-male-cns-v1.0-minconf-0.5.feather'
$neurotransmitters = 'data/body-neurotransmitters-male-cns-v1.0.feather'
$weights = 'data/connectome-weights-male-cns-v1.0-minconf-0.5.feather'
$sensory = 'data/chess_sensory_indices.npy'
$output = 'results/mechanistic_discovery_v1/gain_factorial'
$maxProcessCount = 8 # Start-Process exposes a launcher plus a Python child for each worker.
$gains = @('0.00', '0.02', '0.05', '0.10', '0.20', '0.35', '0.50', '0.75', '1.00', '1.25')
$modes = @('clamped_sensory', 'single_pulse')

function Get-NormalizedGain([string]$gain) {
    return ([double]::Parse($gain, [Globalization.CultureInfo]::InvariantCulture)).ToString('0.##', [Globalization.CultureInfo]::InvariantCulture)
}

function Get-ProgressPath([string]$gain, [string]$mode) {
    $normalized = Get-NormalizedGain $gain
    return Join-Path $output "raw_gain_${normalized}_${mode}.progress.json"
}

function Test-Complete([string]$gain, [string]$mode) {
    $path = Get-ProgressPath $gain $mode
    if (-not (Test-Path -LiteralPath $path)) { return $false }
    try {
        $payload = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
        return $payload.status -eq 'complete' -and [int]$payload.rows -eq 1792
    } catch { return $false }
}

New-Item -ItemType Directory -Force -Path $output | Out-Null
while ($true) {
    $workers = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*run_mechanistic_factorial.py*' })
    foreach ($gain in $gains) {
        foreach ($mode in $modes) {
            if (Test-Complete $gain $mode) { continue }
            if ($workers.Count -ge $maxProcessCount) { break }
            $normalized = Get-NormalizedGain $gain
            $match = $workers | Where-Object {
                $_.CommandLine -like "*run_mechanistic_factorial.py*" -and
                $_.CommandLine -like "*--gain-scale $normalized*" -and
                $_.CommandLine -like "*--input-mode $mode*"
            }
            if ($match) { continue }
            $stem = "raw_gain_${normalized}_${mode}"
            $stdout = Join-Path $output "$stem.stdout.log"
            $stderr = Join-Path $output "$stem.stderr.log"
            $arguments = @(
                'scripts/run_mechanistic_factorial.py', '--checkpoint', $checkpoint,
                '--annotations', $annotations, '--neurotransmitters', $neurotransmitters,
                '--connectome-weights', $weights, '--sensory-indices', $sensory,
                '--output', $output, '--gain-scale', $normalized, '--input-mode', $mode,
                '--resume', '--batch-positions', '4'
            )
            Start-Process -FilePath $python -WorkingDirectory $repo -ArgumentList $arguments -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden
            Start-Sleep -Seconds 2
            $workers = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*run_mechanistic_factorial.py*' })
        }
        if ($workers.Count -ge $maxProcessCount) { break }
    }
    $pending = @($gains | ForEach-Object { $gain = $_; $modes | ForEach-Object { if (-not (Test-Complete $gain $_)) { "$gain/$_" } } })
    if ($pending.Count -eq 0 -and $workers.Count -eq 0) { break }
    Start-Sleep -Seconds 15
}
"factorial queue complete" | Set-Content -LiteralPath (Join-Path $output 'queue.log') -Encoding utf8
