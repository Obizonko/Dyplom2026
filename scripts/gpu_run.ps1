# Етапи для комп'ютера з NVIDIA GPU (преєстрація 1.1). Інструкція: plan/GPU_ETAPY.md
# Запуск з кореня репозиторію (PowerShell):
#   powershell -ExecutionPolicy Bypass -File scripts\gpu_run.ps1 -Stage check
#   powershell -ExecutionPolicy Bypass -File scripts\gpu_run.ps1 -Stage s08
#   powershell -ExecutionPolicy Bypass -File scripts\gpu_run.ps1 -Stage xlmr
#   powershell -ExecutionPolicy Bypass -File scripts\gpu_run.ps1 -Stage all
# Кожен етап можна переривати і запускати знову: готові одиниці (results/.../units/*.json) пропускаються.
param(
    [ValidateSet("check", "s08", "xlmr", "all")] [string]$Stage = "check",
    [int]$Seeds = 5,
    [int]$XlmrSeeds = 3,
    [int]$XlmrPerm = 20,
    [int]$BatchSize = 32,
    [switch]$Unfreeze  # XLM-R: навчати й словникові ембединги (лише якщо GPU має від 12 ГБ; на 6 ГБ не вміщається)
)
$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force logs | Out-Null

function Run-Step([string]$name, [string[]]$pyArgs, [string]$log) {
    Write-Host "[$(Get-Date -Format HH:mm:ss)] $name -> $log"
    & $py -I @pyArgs *> $log
    if ($LASTEXITCODE -ne 0) { throw "$name завершився з кодом $LASTEXITCODE, див. $log" }
}

if ($Stage -in @("check", "all")) {
    & $py -c "import torch; assert torch.cuda.is_available(), 'CUDA недоступна: поставте torch зі збіркою cu128'; print('GPU:', torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/2**30,1), 'ГБ')"
    if ($LASTEXITCODE -ne 0) { throw "немає CUDA" }
    & $py -c "import pyg_lib, torch_geometric; print('PyG', torch_geometric.__version__, 'pyg-lib', pyg_lib.__version__)"
    if ($LASTEXITCODE -ne 0) { throw "немає pyg-lib: див. plan/GPU_ETAPY.md, крок 3" }
    foreach ($f in @("data\raw\telegram_messages.parquet", "data\interim\embeddings\bge_m3_fp16.npy", "data\raw\extracted\parquet_set\forward_edges.parquet",
                     "data\interim\features\copub_pairs.parquet", "data_small\corpus_channel_counts.csv")) {
        if (-not (Test-Path $f)) { throw "немає файлу ${f}: розпакуйте архів даних (plan/GPU_ETAPY.md, крок 2)" }
    }
    Run-Step "v11_prepare (ваги пулу; пари копій рахуються, якщо їх немає)" @("scripts\v11_prepare.py") "logs\gpu_v11_prepare.log"
    Run-Step "S8 sanity: перенавчання на 32 прикладах (GraphSAGE)" @("scripts\s08_run.py", "--sanity", "--arch", "sage") "logs\gpu_s08_sanity_sage.log"
    Run-Step "S8 sanity (R-GCN)" @("scripts\s08_run.py", "--sanity", "--arch", "rgcn") "logs\gpu_s08_sanity_rgcn.log"
    Run-Step "S8 sanity (HGT)" @("scripts\s08_run.py", "--sanity", "--arch", "hgt") "logs\gpu_s08_sanity_hgt.log"
    Run-Step "XLM-R самоперевірка (крихітна модель)" @("scripts\s06_train.py", "--selftest") "logs\gpu_xlmr_selftest.log"
    Write-Host "Перевірку пройдено."
}

if ($Stage -in @("s08", "all")) {
    # Пріоритет 1. GNN за v1.1: повна модель (семантика між каналами, форварди, co_publication, текст bge-m3) і абляції для H2;
    # R-GCN і HGT з текстом і без; 5 seed; P1. Перші 20 одиниць контролю: перемішані мітки для sage_full.
    $variants = "sage_full,sage_nosem,sage_sem975,sage_sem990,sage_nofwd,sage_nocopub,sage_notext,rgcn_full,rgcn_notext,hgt_full,hgt_notext"
    Run-Step "S8 v1.1: GNN, P1, $Seeds seed, 20 перестановок" @("scripts\s08_run.py", "--v11", "--variants", $variants, "--protocols", "P1",
        "--seeds", "$Seeds", "--n-perm", "20", "--out", "results\v11\08_gnn") "logs\v11_s08.log"
}

if ($Stage -in @("xlmr", "all")) {
    # Пріоритет 2. XLM-R base за v1.1 (ваги пулу у втраті), P1 і P2, 3 seed, 3 епохи; контроль: XlmrPerm перестановок по 1 епосі.
    # TF-IDF і bge-m3 у тій самій папці вже пораховані й будуть пропущені.
    $xa = @("scripts\s06_train.py", "--raw-dir", "data\raw", "--splits-dir", "splits", "--emb-dir", "data\interim\embeddings",
        "--out", "results\v11\06_text", "--models", "xlmr", "--protocols", "P1,P2", "--xlmr-protocols", "P1,P2", "--seeds", "$XlmrSeeds",
        "--epochs", "3", "--batch-size", "$BatchSize", "--v11", "--n-perm", "20", "--xlmr-perm", "$XlmrPerm")
    if ($Unfreeze) { $xa += "--unfreeze-embeddings" }
    Run-Step "S6 v1.1: XLM-R" $xa "logs\v11_xlmr.log"
}
Write-Host "Готово. Прогрес: python -I scripts\progress_page.py, далі відкрийте reports\progress.html"
