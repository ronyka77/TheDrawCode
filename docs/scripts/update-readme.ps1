# README Updater
# Description: Updates the main README with new features

function Test-FeatureExists {
    param (
        [string]$content,
        [string]$featureName
    )
    
    return $content -match "### $([regex]::Escape($featureName))"
}

# Main README path
$readmePath = Join-Path $PSScriptRoot "..\..\README.md"

# Check if README exists
if (-not (Test-Path $readmePath)) {
    Write-Host "Error: README.md not found" -ForegroundColor Red
    exit 1
}

# Get feature details
$featureName = Read-Host "Enter feature name"
$featureDesc = Read-Host "Enter feature description"
$docPath = Read-Host "Enter documentation path (e.g., docs/features/feature-name.md)"

# Read current README
$content = Get-Content $readmePath -Raw

# Check if feature already exists
if (Test-FeatureExists $content $featureName) {
    Write-Host "Warning: Feature '$featureName' already exists in README" -ForegroundColor Yellow
    exit 0
}

# Format new feature entry
$newFeature = @"

### $featureName
$featureDesc
[Documentation]($docPath)
"@

# Find or create Features section
if ($content -match '## Features\s*\r?\n') {
    $content = $content -replace '(## Features\s*\r?\n)', "`$1$newFeature"
} else {
    $content += "`n## Features$newFeature"
}

# Update README
$content | Set-Content $readmePath

# Update table of contents if gh-md-toc is available
if (Get-Command gh-md-toc -ErrorAction SilentlyContinue) {
    Write-Host "Updating table of contents..." -ForegroundColor Yellow
    gh-md-toc --insert $readmePath
}

Write-Host "`nREADME updated successfully!" -ForegroundColor Green
Write-Host "Added feature: $featureName" -ForegroundColor Cyan 