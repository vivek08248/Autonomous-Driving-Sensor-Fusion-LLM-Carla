<#
.SYNOPSIS
  Launch CARLA (Low quality) + Ollama (GPU) + drive_in_carla.py   (v20)
  Tuned for: Ryzen 7 7735HS, 16 GB RAM, RTX 3050 Laptop 6 GB.

.DESCRIPTION
  VRAM budget (6 GB): CARLA at -quality-level=Low ~3-3.5 GB, no camera sensors (0 MB),
  qwen2.5:1.5b Q4 ~1.2 GB (qwen2.5:0.5b ~0.5 GB, the size used in the DriveGPT4-V2 paper).
  The old script forced Ollama onto the CPU (CUDA_VISIBLE_DEVICES=""), which made every
  LLM reply take 4-7 s.  Use -OllamaCPU only if CARLA crashes with "out of video memory".

.USAGE
  .\start_drive.ps1                                   # AI mode, interactive
  .\start_drive.ps1 -Scenario TC_47_RED_LIGHT_STOP
  .\start_drive.ps1 -Batch                            # all 100 test cases -> report
  .\start_drive.ps1 -Batch -From 27 -To 54 -Model qwen2.5:0.5b
  .\start_drive.ps1 -Batch -LLM off                   # expert-only baseline
  .\start_drive.ps1 -Mode MANUAL
  .\start_drive.ps1 -Demo                             # TOUR for the guide: all test cases one after another,
                                                      #   3 min each, loops forever, ALL keys work (M L N P R C T G B)
  .\start_drive.ps1 -Demo -Duration 90 -From 27 -To 54
  .\start_drive.ps1 -Batch -Demo                      # same length, but runs each case once and writes the report
  .\start_drive.ps1 -Batch -Demo -Duration 120 -From 47 -To 54
  .\start_drive.ps1 -Batch -Fast                      # quick metrics (faster than real time)
#>
param(
    [string]$Model = "qwen2.5:1.5b",
    [ValidateSet("AI","MANUAL")] [string]$Mode = "AI",
    [string]$Scenario = "",
    [switch]$Batch,
    [int]$From = 1,
    [int]$To = 999,
    [ValidateSet("on","off")] [string]$LLM = "on",
    [ValidateSet("dagger","advisory","expert")] [string]$Authority = "dagger",
    [switch]$OllamaCPU,
    [switch]$Demo,
    [switch]$Fast,
    [int]$Duration = 0,
    [double]$Speed = 1.0,
    [switch]$NoCarlaStart
)

$ErrorActionPreference = "Continue"
$ProjectDir = $PSScriptRoot
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

function Test-Port($p) {
    try { $c = New-Object System.Net.Sockets.TcpClient; $c.Connect("127.0.0.1", $p); $c.Close(); return $true }
    catch { return $false }
}

# ---------------------------------------------------------------- 1. Ollama
if ($LLM -eq "on") {
    Write-Host "[1/4] Ollama ..." -ForegroundColor Cyan
    $env:OLLAMA_KEEP_ALIVE = "30m"
    $env:OLLAMA_NUM_PARALLEL = "1"          # one request at a time -> lowest latency
    $env:OLLAMA_MAX_LOADED_MODELS = "1"     # never keep two models in 6 GB VRAM
    $env:OLLAMA_FLASH_ATTENTION = "1"
    if ($OllamaCPU) { $env:CUDA_VISIBLE_DEVICES = "-1"; Write-Host "  CPU mode requested" -ForegroundColor Yellow }
    else { Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue }

    if (-not (Test-Port 11434)) {
        $ollamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
        if (-not (Test-Path $ollamaExe)) { $ollamaExe = "ollama" }
        Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden
        for ($i = 0; $i -lt 15 -and -not (Test-Port 11434); $i++) { Start-Sleep -Seconds 1 }
    } else {
        Write-Host "  Ollama already running (if it was started in CPU mode, quit it from the tray and re-run)." -ForegroundColor DarkGray
    }
    try {
        $tags = (Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5).models.name
        if ($tags -notcontains $Model) {
            Write-Host "  pulling $Model ..." -ForegroundColor Yellow
            & ollama pull $Model
        }
        Write-Host "  warming $Model ..." -ForegroundColor DarkGray
        $body = @{ model = $Model; prompt = "ok"; stream = $false; keep_alive = "30m"; options = @{ num_predict = 1; num_ctx = 1024 } } | ConvertTo-Json -Depth 3
        Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 120 | Out-Null
        Write-Host "  model ready" -ForegroundColor Green
        & ollama ps
    } catch { Write-Host "  Ollama not reachable -> the expert planner will drive alone." -ForegroundColor Yellow }
}

# ---------------------------------------------------------------- 2. CARLA
if (-not $NoCarlaStart -and -not (Test-Port 2000)) {
    Write-Host "[2/4] CARLA (Low quality) ..." -ForegroundColor Cyan
    Start-Process -FilePath "$ProjectDir\CarlaUE4.exe" -WorkingDirectory $ProjectDir `
        -ArgumentList "-quality-level=Low", "-carla-rpc-port=2000", "-nosound", "-ResX=1280", "-ResY=720", "-windowed"
    for ($i = 0; $i -lt 36 -and -not (Test-Port 2000); $i++) { Write-Host "  waiting for CARLA ($($i+1)/36)"; Start-Sleep -Seconds 5 }
    if (-not (Test-Port 2000)) { Write-Host "  ERROR: CARLA did not start" -ForegroundColor Red; exit 1 }
    Start-Sleep -Seconds 8     # let the map finish streaming
} else { Write-Host "[2/4] CARLA already running" -ForegroundColor DarkGray }

# ---------------------------------------------------------------- 3. Driver
Write-Host "[3/4] drive_in_carla.py" -ForegroundColor Cyan
$env:PYTHONPATH = "$ProjectDir;$ProjectDir\PythonAPI\carla;$ProjectDir\PythonAPI"
Set-Location $ProjectDir
$a = @("drive_in_carla.py", "--model", $Model, "--mode", $Mode, "--llm", $LLM, "--authority", $Authority)
if ($Scenario) { $a += @("--scenario", $Scenario) }
if ($Batch)    { $a += @("--batch") }
if ($Batch -or $Demo) { $a += @("--from", "$From", "--to", "$To") }
if ($Demo)     { $a += @("--demo") }
if ($Fast)     { $a += @("--fast") }
if ($Duration -gt 0) { $a += @("--duration", "$Duration") }
if ($Speed -ne 1.0)  { $a += @("--speed", "$Speed") }
Write-Host "  ESC quit | M AI/Manual | N/P test case | R restart | L model | C view | T route | G pause tour | B assist"
& $Python @a

Write-Host "[4/4] done. Report: LLM_Comparison_Report.md" -ForegroundColor Green
