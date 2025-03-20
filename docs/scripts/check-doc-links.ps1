# Documentation Link Checker
# Description: Validates documentation links and references

function Test-MarkdownLinks {
    param (
        [string]$filePath
    )
    
    if (-not (Test-Path $filePath)) {
        Write-Host "Error: File not found - $filePath" -ForegroundColor Red
        return $false
    }
    
    $content = Get-Content $filePath -Raw
    $linkPattern = '\[([^\]]+)\]\(([^)]+)\)'
    $matches = [regex]::Matches($content, $linkPattern)
    $hasErrors = $false
    
    foreach ($match in $matches) {
        $linkText = $match.Groups[1].Value
        $linkPath = $match.Groups[2].Value
        
        # Skip external links and anchors
        if ($linkPath -match '^(http|https|#)') {
            continue
        }
        
        # Convert mdc: links to relative paths
        if ($linkPath -match '^mdc:') {
            $linkPath = $linkPath -replace '^mdc:', ''
        }
        
        # Resolve relative path
        $resolvedPath = Join-Path (Split-Path $filePath) $linkPath
        $resolvedPath = [System.IO.Path]::GetFullPath($resolvedPath)
        
        if (-not (Test-Path $resolvedPath)) {
            Write-Host "Broken link in $filePath" -ForegroundColor Red
            Write-Host "  Link text: $linkText" -ForegroundColor Yellow
            Write-Host "  Target: $linkPath" -ForegroundColor Yellow
            $hasErrors = $true
        }
    }
    
    return -not $hasErrors
}

# Main documentation paths
$docsRoot = Join-Path $PSScriptRoot "..\..\"
$docFiles = @(
    "README.md",
    "CHANGELOG.md",
    "DOCS-CHANGELOG.md",
    "docs\architecture\*.md",
    "docs\guides\*.md"
)

Write-Host "Checking Documentation Links`n" -ForegroundColor Green
$allValid = $true

foreach ($pattern in $docFiles) {
    $fullPattern = Join-Path $docsRoot $pattern
    $files = Get-ChildItem $fullPattern -ErrorAction SilentlyContinue
    
    foreach ($file in $files) {
        Write-Host "Checking $($file.Name)..." -ForegroundColor Cyan
        if (-not (Test-MarkdownLinks $file.FullName)) {
            $allValid = $false
        }
    }
}

if ($allValid) {
    Write-Host "`nAll documentation links are valid!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`nFound broken documentation links." -ForegroundColor Red
    exit 1
} 