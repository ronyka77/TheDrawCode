# Fetch Project Context
# Description: Gathers context from project plans

function Get-PlanContent {
    param (
        [string]$filePath
    )
    
    if (Test-Path $filePath) {
        Write-Host "Reading $filePath..." -ForegroundColor Yellow
        Get-Content $filePath
    } else {
        Write-Host "Warning: $filePath not found" -ForegroundColor Red
    }
}

# Main paths
$podcastPlanPath = Join-Path $PSScriptRoot "..\..\docs\plan-podcast.md"

Write-Host "Gathering Project Context`n" -ForegroundColor Green

# Fetch podcast plan
Write-Host "`nPodcast Feature Plan:" -ForegroundColor Cyan
Get-PlanContent $podcastPlanPath

Write-Host "`nContext gathering complete!" -ForegroundColor Green 