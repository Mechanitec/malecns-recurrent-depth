param(
    [int]$ShardCount = 4
)

$ErrorActionPreference = 'Stop'
$python = 'C:\15. Project Shakhsi\Research\3. AI\maleCNS-recurrent-depth\.venv\Scripts\python.exe'
$root = Split-Path -Parent $PSScriptRoot
$results = Join-Path $root 'results\population_study'
$log = Join-Path $results 'plan2_pipeline.log'

function Invoke-Step {
    param([string]$Name, [string[]]$Arguments)
    Add-Content -LiteralPath $log -Value ("[$(Get-Date -Format s)] START $Name")
    & $python @Arguments *>> $log
    if ($LASTEXITCODE -ne 0) {
        Add-Content -LiteralPath $log -Value ("[$(Get-Date -Format s)] FAIL $Name exit=$LASTEXITCODE")
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    Add-Content -LiteralPath $log -Value ("[$(Get-Date -Format s)] COMPLETE $Name")
}

while ($true) {
    $complete = 0
    for ($index = 0; $index -lt $ShardCount; $index++) {
        $metadataPath = Join-Path $results ('shards\shard_{0:D3}\activation_shard_metadata.json' -f ($index * 64))
        if (Test-Path -LiteralPath $metadataPath) {
            $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
            if ($metadata.status -eq 'complete') { $complete++ }
        }
    }
    if ($complete -eq $ShardCount) { break }
    Add-Content -LiteralPath $log -Value ("[$(Get-Date -Format s)] WAIT shards=$complete/$ShardCount")
    Start-Sleep -Seconds 60
}

Invoke-Step 'merge activation shards' @('scripts/merge_population_activation_shards.py','--corpus','data/chess_development_corpus_v1.csv','--manifests','data/populations_v2/manifests','--shards','results/population_study/shards','--output','results/population_study/_activation_cache','--limit-positions','256')
Invoke-Step 'fit brain-region decoding atlas' @('scripts/fit_brain_region_decoding.py','--corpus','data/chess_development_corpus_v1.csv','--manifests','data/populations_v2/manifests','--cache','results/population_study/_activation_cache','--output','results/population_study')
Invoke-Step 'depth probe transfer' @('scripts/run_depth_probe_transfer_population.py','--corpus','data/chess_development_corpus_v1.csv','--manifests','data/populations_v2/manifests','--cache','results/population_study/_activation_cache','--output','results/population_study')
Invoke-Step 'bypass baselines' @('scripts/run_bypass_baselines.py','--checkpoint','results/training/v1_depth16_rate_large/readout_checkpoint.npz','--annotations','data/body-annotations-male-cns-v1.0-minconf-0.5.feather','--neurotransmitters','data/body-neurotransmitters-male-cns-v1.0.feather','--weights','data/connectome-weights-male-cns-v1.0-minconf-0.5.feather','--sensory-indices','data/chess_sensory_indices.npy','--corpus','data/chess_development_corpus_v1.csv','--output','results/population_study')
Invoke-Step 'input population study' @('scripts/run_input_population_study.py','--checkpoint','results/training/v1_depth16_rate_large/readout_checkpoint.npz','--annotations','data/body-annotations-male-cns-v1.0-minconf-0.5.feather','--neurotransmitters','data/body-neurotransmitters-male-cns-v1.0.feather','--weights','data/connectome-weights-male-cns-v1.0-minconf-0.5.feather','--sensory-indices','data/chess_sensory_indices.npy','--corpus','data/chess_development_corpus_v1.csv','--manifests','data/populations_v2/manifests','--output','results/population_study','--batch-positions','2')
Invoke-Step 'H-BIO-1 study' @('scripts/run_h_bio_1.py','--checkpoint','results/training/v1_depth16_rate_large/readout_checkpoint.npz','--annotations','data/body-annotations-male-cns-v1.0-minconf-0.5.feather','--neurotransmitters','data/body-neurotransmitters-male-cns-v1.0.feather','--weights','data/connectome-weights-male-cns-v1.0-minconf-0.5.feather','--sensory-indices','data/chess_sensory_indices.npy','--corpus','data/chess_development_corpus_v1.csv','--manifests','data/populations_v2/manifests','--output','results/population_study','--batch-positions','2')
Invoke-Step 'population size scaling' @('scripts/run_population_size_scaling.py','--checkpoint','results/training/v1_depth16_rate_large/readout_checkpoint.npz','--annotations','data/body-annotations-male-cns-v1.0-minconf-0.5.feather','--neurotransmitters','data/body-neurotransmitters-male-cns-v1.0.feather','--weights','data/connectome-weights-male-cns-v1.0-minconf-0.5.feather','--sensory-indices','data/chess_sensory_indices.npy','--corpus','data/chess_development_corpus_v1.csv','--manifests','data/populations_v2/manifests','--atlas','results/population_study/malecns_region_atlas.json','--output','results/population_study','--batch-positions','2')
Invoke-Step 'seed robustness' @('scripts/run_seed_robustness.py','--checkpoint','results/training/v1_depth16_rate_large/readout_checkpoint.npz','--annotations','data/body-annotations-male-cns-v1.0-minconf-0.5.feather','--neurotransmitters','data/body-neurotransmitters-male-cns-v1.0.feather','--weights','data/connectome-weights-male-cns-v1.0-minconf-0.5.feather','--sensory-indices','data/chess_sensory_indices.npy','--readout-indices','data/chess_readout_indices.npy','--manifests','data/populations_v2/manifests','--corpus','data/chess_development_corpus_v1.csv','--output','results/population_study','--seeds','20','--positions-per-split','32','--batch-positions','2')
Invoke-Step 'topology-matched manifests' @('scripts/build_topology_matched_manifests.py','--annotations','data/body-annotations-male-cns-v1.0-minconf-0.5.feather','--neurotransmitters','data/body-neurotransmitters-male-cns-v1.0.feather','--weights','data/connectome-weights-male-cns-v1.0-minconf-0.5.feather','--sensory-indices','data/chess_sensory_indices.npy','--readout-indices','data/chess_readout_indices.npy','--atlas','results/population_study/malecns_region_atlas.json','--output','data/populations_v2/manifests')
Invoke-Step 'graph predictors' @('scripts/build_graph_predictors.py','--annotations','data/body-annotations-male-cns-v1.0-minconf-0.5.feather','--neurotransmitters','data/body-neurotransmitters-male-cns-v1.0.feather','--weights','data/connectome-weights-male-cns-v1.0-minconf-0.5.feather','--sensory-indices','data/chess_sensory_indices.npy','--readout-indices','data/chess_readout_indices.npy','--atlas','results/population_study/malecns_region_atlas.json','--manifests','data/populations_v2/manifests','--output','results/population_study')
Invoke-Step 'derived population artifacts' @('scripts/build_population_artifacts.py','--root','results/population_study')
Invoke-Step 'fresh confirmation' @('scripts/run_fresh_confirmation.py','--checkpoint','results/training/v1_depth16_rate_large/readout_checkpoint.npz','--annotations','data/body-annotations-male-cns-v1.0-minconf-0.5.feather','--neurotransmitters','data/body-neurotransmitters-male-cns-v1.0.feather','--weights','data/connectome-weights-male-cns-v1.0-minconf-0.5.feather','--sensory-indices','data/chess_sensory_indices.npy','--corpus','data/chess_confirmatory_corpus_v2.csv','--manifests','data/populations_v2/manifests','--selection','results/population_study/final_selection.json','--output','results/population_study','--batch-positions','2')
Add-Content -LiteralPath $log -Value ("[$(Get-Date -Format s)] PLAN2 COMPLETE")
