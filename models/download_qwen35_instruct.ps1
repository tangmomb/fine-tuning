$ErrorActionPreference = 'Stop'

# Keep every Hugging Face cache and temporary download inside this repository.
$projectRoot = Split-Path -Parent $PSScriptRoot
$hfRoot = Join-Path $projectRoot '.hf'
$env:HF_HOME = $hfRoot
$env:HF_HUB_CACHE = Join-Path $hfRoot 'hub'
$env:HF_ASSETS_CACHE = Join-Path $hfRoot 'assets'
$env:HF_XET_CACHE = Join-Path $hfRoot 'xet'
$env:TRANSFORMERS_CACHE = Join-Path $hfRoot 'transformers'
$env:HF_HUB_DISABLE_XET = '1'

$models = @(
    @{ Repository = 'Qwen/Qwen3.5-0.8B'; Directory = 'Qwen3.5-0.8B' },
    @{ Repository = 'Qwen/Qwen3.5-2B'; Directory = 'Qwen3.5-2B' },
    @{ Repository = 'Qwen/Qwen3.5-4B'; Directory = 'Qwen3.5-4B' },
    @{ Repository = 'Qwen/Qwen3.5-9B'; Directory = 'Qwen3.5-9B' }
)

foreach ($model in $models) {
    $target = Join-Path $PSScriptRoot $model.Directory
    Write-Host "Downloading $($model.Repository) to $target"
    hf download $model.Repository --local-dir $target --max-workers 4
}
