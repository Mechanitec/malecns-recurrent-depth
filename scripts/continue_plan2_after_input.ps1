param(
    [int]$BatchPositions = 8,
    [int]$PollSeconds = 60
)

$ErrorActionPreference = 'Stop'
$python = 'C:\15. Project Shakhsi\Research\3. AI\maleCNS-recurrent-depth\.venv\Scripts\python.exe'
$root = Split-Path -Parent $PSScriptRoot
$results = Join-Path $root 'results\population_study'
$log = Join-Path $results 'plan2_continuation.log'

Set-Location $root

function Write-Log {
    param([string]$Message)
    Add-Content -LiteralPath $log -Value ("[$(Get-Date -Format s)] $Message")
}

function Test-CompleteMetadata {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    try {
        return ((Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json).status -eq 'complete')
    } catch {
        return $false
    }
}

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [string]$CompletionMetadata
    )
    if (Test-CompleteMetadata $CompletionMetadata) {
        Write-Log "SKIP $Name (already complete)"
        return
    }
    Write-Log "START $Name"
    & $python @Arguments *>> $log
    if ($LASTEXITCODE -ne 0) {
        Write-Log "FAIL $Name exit=$LASTEXITCODE"
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    if (-not (Test-CompleteMetadata $CompletionMetadata)) {
        Write-Log "FAIL $Name missing completion metadata"
        throw "$Name did not produce complete metadata"
    }
    Write-Log "COMPLETE $Name"
}

Write-Log 'WAIT input population study'
while (-not (Test-CompleteMetadata (Join-Path $results 'input_population_metadata.json'))) {
    $running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'run_input_population_study' }
    if (-not $running) {
        Write-Log 'FAIL input population study stopped before completion'
        throw 'input population study stopped before producing complete metadata'
    }
    Start-Sleep -Seconds $PollSeconds
}
Write-Log 'COMPLETE input population study'

Invoke-Step 'H-BIO-1 study' @('scripts/run_h_bio_1.py','--checkpoint','results/training/v1_depth16_rate_large/readout_checkpoint.npz','--annotations','data/body-annotations-male-cns-v1.0-minconf-0.5.feather','--neurotransmitters','data/body-neurotransmitters-male-cns-v1.0.feather','--weights','data/connectome-weights-male-cns-v1.0-minconf-0.5.feather','--sensory-indices','data/chess_sensory_indices.npy','--corpus','data/chess_development_corpus_v1.csv','--manifests','data/populations_v2/manifests','--output','results/population_study','--batch-positions',$BatchPositions) 'results/population_study/h_bio_1_metadata.json'
& $python scripts/build_population_artifacts.py --root results/population_study --manifests data/populations_v2/manifests *>> $log
if ($LASTEXITCODE -ne 0) { Write-Log "FAIL derived population artifacts exit=$LASTEXITCODE"; throw 'derived population artifacts failed' }
if (-not (Test-Path -LiteralPath (Join-Path $results 'final_selection.json'))) { Write-Log 'FAIL final selection missing'; throw 'final selection was not produced' }
Write-Log 'COMPLETE derived population artifacts and frozen rule'

Invoke-Step 'population size scaling' @('scripts/run_population_size_scaling.py','--checkpoint','results/training/v1_depth16_rate_large/readout_checkpoint.npz','--annotations','data/body-annotations-male-cns-v1.0-minconf-0.5.feather','--neurotransmitters','data/body-neurotransmitters-male-cns-v1.0.feather','--weights','data/connectome-weights-male-cns-v1.0-minconf-0.5.feather','--sensory-indices','data/chess_sensory_indices.npy','--corpus','data/chess_development_corpus_v1.csv','--manifests','data/populations_v2/manifests','--atlas','results/population_study/malecns_region_atlas.json','--output','results/population_study','--batch-positions',$BatchPositions) 'results/population_study/population_size_scaling_metadata.json'
Invoke-Step 'seed robustness' @('scripts/run_seed_robustness.py','--checkpoint','results/training/v1_depth16_rate_large/readout_checkpoint.npz','--annotations','data/body-annotations-male-cns-v1.0-minconf-0.5.feather','--neurotransmitters','data/body-neurotransmitters-male-cns-v1.0.feather','--weights','data/connectome-weights-male-cns-v1.0-minconf-0.5.feather','--sensory-indices','data/chess_sensory_indices.npy','--readout-indices','data/chess_readout_indices.npy','--manifests','data/populations_v2/manifests','--corpus','data/chess_development_corpus_v1.csv','--output','results/population_study','--seeds','20','--positions-per-split','32','--batch-positions',$BatchPositions) 'results/population_study/seed_robustness_metadata.json'
Invoke-Step 'fresh confirmation' @('scripts/run_fresh_confirmation.py','--checkpoint','results/training/v1_depth16_rate_large/readout_checkpoint.npz','--annotations','data/body-annotations-male-cns-v1.0-minconf-0.5.feather','--neurotransmitters','data/body-neurotransmitters-male-cns-v1.0.feather','--weights','data/connectome-weights-male-cns-v1.0-minconf-0.5.feather','--sensory-indices','data/chess_sensory_indices.npy','--corpus','data/chess_confirmatory_corpus_v2.csv','--manifests','data/populations_v2/manifests','--selection','results/population_study/final_selection.json','--output','results/population_study','--batch-positions',$BatchPositions) 'results/population_study/final_confirmation_summary.json'
Write-Log 'PLAN2 CONTINUATION COMPLETE'
