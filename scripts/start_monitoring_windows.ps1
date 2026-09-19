<#
.SYNOPSIS
    Installs (portable zip) and launches Prometheus & Grafana locally on Windows without needing Docker.

.DESCRIPTION
    1. Downloads portable standalone Windows releases of Prometheus & Grafana if not already present.
    2. Configures Prometheus to use monitoring/prometheus/prometheus.yml.
    3. Copies provisioning configs (datasource + dashboard) into Grafana's conf directory
       with paths fixed for local (non-Docker) use.
    4. Launches Prometheus (:9090) and Grafana (:3000) in separate terminal windows.
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

# 3. Set up Grafana provisioning for local use
Write-Host "`nConfiguring Grafana provisioning..." -ForegroundColor Yellow

# Create provisioning directories inside Grafana
$grafProvDS = Join-Path $grafanaDir "conf\provisioning\datasources"
$grafProvDB = Join-Path $grafanaDir "conf\provisioning\dashboards"
$grafDashDir = Join-Path $grafanaDir "dashboards"

New-Item -ItemType Directory -Path $grafProvDS -Force | Out-Null
New-Item -ItemType Directory -Path $grafProvDB -Force | Out-Null
New-Item -ItemType Directory -Path $grafDashDir -Force | Out-Null

# Write datasource config pointing to localhost (not Docker hostname)
$datasourceYml = @"
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
    editable: true
    jsonData:
      timeInterval: 2s
      httpMethod: POST
"@
Set-Content -Path (Join-Path $grafProvDS "datasource.yml") -Value $datasourceYml -Encoding UTF8

# Write dashboard provisioning config with local path
$dashboardPath = $grafDashDir -replace '\\', '/'
$dashboardYml = @"
apiVersion: 1

providers:
  - name: "CloudServe Dashboards"
    orgId: 1
    folder: "CloudServe"
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: $dashboardPath
"@
Set-Content -Path (Join-Path $grafProvDB "dashboards.yml") -Value $dashboardYml -Encoding UTF8

# Copy the actual dashboard JSON
$srcDashboard = Join-Path $rootDir "monitoring\grafana\dashboards\cloudserve_support_system.json"
if (Test-Path $srcDashboard) {
    Copy-Item -Path $srcDashboard -Destination $grafDashDir -Force
    Write-Host "Dashboard JSON copied." -ForegroundColor Green
} else {
    Write-Host "WARNING: Dashboard JSON not found at $srcDashboard" -ForegroundColor Red
}

Write-Host "Provisioning configured." -ForegroundColor Green

if ($DownloadOnly) {
    Write-Host "Downloads complete." -ForegroundColor Green
    exit 0
}

# 4. Kill any existing Prometheus/Grafana processes
Get-Process -Name "prometheus" -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process -Name "grafana-server" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

# 5. Launch Prometheus
$promConfig = Join-Path $rootDir "monitoring\prometheus\prometheus.yml"
Write-Host "`nStarting Prometheus on http://localhost:9090..." -ForegroundColor Yellow
Start-Process -FilePath $promExe -ArgumentList "--config.file=`"$promConfig`" --web.listen-address=0.0.0.0:9090"

# 6. Launch Grafana
Write-Host "Starting Grafana on http://localhost:3000..." -ForegroundColor Yellow
$grafanaHome = $grafanaDir
Start-Process -FilePath $grafanaExe -WorkingDirectory $grafanaDir -ArgumentList "--homepath=`"$grafanaHome`""

# 7. Wait and verify
Write-Host "`nWaiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host "`n========================================" -ForegroundColor Green
Write-Host " Monitoring stack is running!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host " - Prometheus UI:     http://localhost:9090" -ForegroundColor Cyan
Write-Host " - Grafana Dashboard: http://localhost:3000 (admin / admin)" -ForegroundColor Cyan
Write-Host ""
Write-Host "IMPORTANT: Start your FastAPI app first so metrics appear:" -ForegroundColor Yellow
Write-Host "  python -m uvicorn src.api:app --host 0.0.0.0 --port 8000" -ForegroundColor White
Write-Host ""
Write-Host "Then generate traffic:" -ForegroundColor Yellow
Write-Host "  python scripts/simulate_traffic.py --count 20 --delay 2.0" -ForegroundColor White
