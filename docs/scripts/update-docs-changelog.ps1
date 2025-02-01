# Documentation Changelog Updater
# Description: Updates the documentation changelog with new entries

function Get-NewVersion {
    param (
        [string]$currentVersion
    )
    
    Write-Host "Current version: $currentVersion" -ForegroundColor Yellow
    Write-Host "Select version increment type:"
    Write-Host "1) Major (x.0.0)"
    Write-Host "2) Minor (0.x.0)"
    Write-Host "3) Patch (0.0.x)"
    
    $choice = Read-Host "Enter choice (1-3)"
    $version = $currentVersion -split '\.'
    
    switch ($choice) {
        "1" { $version[0] = [int]$version[0] + 1; $version[1] = "0"; $version[2] = "0" }
        "2" { $version[1] = [int]$version[1] + 1; $version[2] = "0" }
        "3" { $version[2] = [int]$version[2] + 1 }
        default { 
            Write-Host "Invalid choice. Using patch increment." -ForegroundColor Red
            $version[2] = [int]$version[2] + 1
        }
    }
    
    return $version -join "."
}

# Main changelog path
$changelogPath = Join-Path $PSScriptRoot "..\..\DOCS-CHANGELOG.md"

# Read current version
if (Test-Path $changelogPath) {
    $content = Get-Content $changelogPath
    $versionLine = $content | Where-Object { $_ -match '## \[(\d+\.\d+\.\d+)\]' } | Select-Object -First 1
    if ($versionLine -match '\[(\d+\.\d+\.\d+)\]') {
        $currentVersion = $matches[1]
    } else {
        $currentVersion = "0.1.0"
    }
} else {
    $currentVersion = "0.1.0"
    "# Documentation Changelog`n" | Set-Content $changelogPath
}

# Get new version
$newVersion = Get-NewVersion $currentVersion

# Get change description
$changeDesc = Read-Host "Enter change description"

# Format new entry
$date = Get-Date -Format "yyyy-MM-dd"
$newEntry = @"

## [$newVersion] - $date
### Changed
- $changeDesc
"@

# Update changelog
$content = Get-Content $changelogPath
$content[0] = $content[0] + $newEntry
$content | Set-Content $changelogPath

Write-Host "`nChangelog updated successfully!" -ForegroundColor Green
Write-Host "New version: $newVersion" -ForegroundColor Cyan 