# Levanta todo el sistema con una sola instruccion (Windows / PowerShell):
#   .\run.ps1          -> construye y levanta Redis 8 + Chroma + API
#   .\run.ps1 logs     -> sigue los logs de la API
#   .\run.ps1 down     -> apaga todo (los datos quedan en los volumenes)
param([string]$Accion = "up")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Accion -eq "logs") { docker compose logs -f api; exit }
if ($Accion -eq "down") { docker compose down; exit }

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Falta Docker Desktop: https://docs.docker.com/desktop/" -ForegroundColor Red; exit 1
}
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Se creo .env a partir de .env.example. Completa OPENAI_API_KEY (y LANGSMITH_API_KEY) y volve a correr .\run.ps1" -ForegroundColor Yellow
    exit 1
}
if (Select-String -Path .env -Pattern '^OPENAI_API_KEY=(sk-\.\.\.)?$' -Quiet) {
    Write-Host "OPENAI_API_KEY en .env todavia tiene el valor de ejemplo." -ForegroundColor Red; exit 1
}

Write-Host "Construyendo y levantando Redis 8 + Chroma + API..." -ForegroundColor Cyan
# --force-recreate api: si cambiaste el .env, el contenedor lo vuelve a leer.
docker compose up --build -d --force-recreate api
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host -NoNewline "Esperando a la API (la primera vez indexa el corpus)"
for ($i = 0; $i -lt 90; $i++) {
    $caido = docker compose ps -a -q --status exited --status restarting api
    if ($caido) {
        Write-Host ""
        Write-Host "La API se detuvo al arrancar. Motivo (ultimas lineas del log):" -ForegroundColor Red
        docker compose logs --tail 12 api
        exit 1
    }
    try {
        $salud = Invoke-RestMethod -Uri http://localhost:8000/health -TimeoutSec 3
        Write-Host ""
        Write-Host "Sistema listo:" -ForegroundColor Green
        $salud | ConvertTo-Json -Depth 4
        Write-Host ""
        Write-Host "   Documentacion interactiva:  http://localhost:8000/docs"
        Write-Host "   5 pruebas de punta a punta: python scripts/demo_escenarios.py"
        Write-Host "   Carga concurrente:          python scripts/carga_concurrente.py"
        Write-Host "   Logs:                       .\run.ps1 logs"
        exit 0
    } catch {
        Write-Host -NoNewline "."
        Start-Sleep -Seconds 2
    }
}
Write-Host ""
Write-Host "La API no respondio a tiempo. Ultimos logs:" -ForegroundColor Red
docker compose logs --tail 40 api
exit 1
