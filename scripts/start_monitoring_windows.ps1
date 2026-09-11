<#
.SYNOPSIS
    Installs (portable zip) and launches Prometheus & Grafana locally on Windows without needing Docker.

.DESCRIPTION
    1. Downloads portable standalone Windows releases of Prometheus & Grafana if not already present.
    2. Configures Prometheus to use monitoring/prometheus/prometheus.yml.
    3. Launches Prometheus (:9090) and Grafana (:3000) in separate terminal windows.
#>

param (
    [switch]$DownloadOnly
)

$ErrorActionPreference = "Stop"
$rootDir = (Get-Item $PSScriptRoot).Parent.FullName
$toolsDir = Join-Path $rootDir "monitoring\tools"

if (-not (Test-Path $toolsDir)) {
    New-Item -ItemType Directory -Path $toolsDir | Out-Null
}

$promVersion = "2.51.0"
$promDir = Join-Path $toolsDir "prometheus-$promVersion.windows-amd64"
$promZip = Join-Path $toolsDir "prometheus.zip"
$promExe = Join-Path $promDir "prometheus.exe"

$grafanaVersion = "10.4.0"
$grafanaDir = Join-Path $toolsDir "grafana-v$grafanaVersion"
$grafanaZip = Join-Path $toolsDir "grafana.zip"
$grafanaExe = Join-Path $grafanaDir "bin\grafana-server.exe"

# 1. Download & Extract Prometheus if missing
if (-not (Test-Path $promExe)) {
    Write-Host "[1/2] Downloading Prometheus $promVersion for Windows..." -ForegroundColor Cyan
    $promUrl = "https://github.com/prometheus/prometheus/releases/download/v$promVersion/prometheus-$promVersion.windows-amd64.zip"
    Invoke-WebRequest -Uri $promUrl -OutFile $promZip
    Write-Host "Extracting Prometheus..." -ForegroundColor Cyan
    Expand-Archive -Path $promZip -DestinationPath $toolsDir -Force
    Remove-Item $promZip -Force
    Write-Host "Prometheus ready at $promExe" -ForegroundColor Green
} else {
    Write-Host "Prometheus is already downloaded." -ForegroundColor Green
}

# 2. Download & Extract Grafana if missing
if (-not (Test-Path $grafanaExe)) {
    Write-Host "[2/2] Downloading Grafana $grafanaVersion for Windows..." -ForegroundColor Cyan
    $grafanaUrl = "https://dl.grafana.com/oss/release/grafana-$grafanaVersion.windows-amd64.zip"
    Invoke-WebRequest -Uri $grafanaUrl -OutFile $grafanaZip
    Write-Host "Extracting Grafana..." -ForegroundColor Cyan
    Expand-Archive -Path $grafanaZip -DestinationPath $toolsDir -Force
    Remove-Item $grafanaZip -Force
    Write-Host "Grafana ready at $grafanaExe" -ForegroundColor Green
} else {
    Write-Host "Grafana is already downloaded." -ForegroundColor Green
}

if ($DownloadOnly) {
    Write-Host "Downloads complete." -ForegroundColor Green
    exit 0
}

# 3. Launch Prometheus
$promConfig = Join-Path $rootDir "monitoring\prometheus\prometheus.yml"
Write-Host "`nStarting Prometheus on http://localhost:9090..." -ForegroundColor Yellow
Start-Process -FilePath $promExe -ArgumentList "--config.file=`"$promConfig`" --web.listen-address=0.0.0.0:9090"

# 4. Launch Grafana
Write-Host "Starting Grafana on http://localhost:3000..." -ForegroundColor Yellow
$grafanaHome = $grafanaDir
Start-Process -FilePath $grafanaExe -WorkingDirectory $grafanaDir -ArgumentList "--homepath=`"$grafanaHome`""

Write-Host "`nMonitoring stack is running!" -ForegroundColor Green
Write-Host " - Prometheus UI:     http://localhost:9090" -ForegroundColor Cyan
Write-Host " - Grafana Dashboard: http://localhost:3000 (admin / admin)" -ForegroundColor Cyan
Write-Host "`nYou can now run:" -ForegroundColor White
Write-Host "  python scripts/simulate_traffic.py --count 20 --delay 2.0" -ForegroundColor Yellow
