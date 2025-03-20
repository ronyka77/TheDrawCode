# Feature Documentation Generator
# Usage: .\Create-FeatureDocs.ps1 -FeatureName <feature-name>
# Description: Creates standardized documentation structure for new features

param(
    [Parameter(Mandatory=$true)]
    [string]$FeatureName
)

# Function to validate feature name
function Test-FeatureName {
    param([string]$Name)
    
    if ($Name -notmatch '^[a-zA-Z][a-zA-Z0-9_-]*$') {
        Write-Host "Error: Feature name must start with a letter and contain only letters, numbers, hyphens, or underscores" -ForegroundColor Red
        exit 1
    }
}

# Validate feature name
Test-FeatureName $FeatureName

# Setup directories
$DocsDir = Join-Path $PSScriptRoot "..\..\features\$FeatureName"
$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# Create feature documentation directory structure
Write-Host "Creating documentation structure for feature: $FeatureName" -ForegroundColor Green

# Create directories
$Directories = @(
    $DocsDir,
    "$DocsDir\assets",
    "$DocsDir\examples",
    "$DocsDir\tests"
)

foreach ($Dir in $Directories) {
    if (-not (Test-Path $Dir)) {
        New-Item -ItemType Directory -Path $Dir | Out-Null
        Write-Host "Created directory: $Dir" -ForegroundColor Cyan
    }
}

# Template files to create
$Templates = @{
    "README.md" = "Feature overview and quick start"
    "architecture.md" = "Technical architecture and design decisions"
    "components.md" = "Component documentation and interactions"
    "api.md" = "API endpoints and specifications"
    "testing.md" = "Test cases and validation criteria"
    "deployment.md" = "Deployment instructions and configuration"
    "monitoring.md" = "Monitoring and maintenance guidelines"
}

# Create documentation files
foreach ($Template in $Templates.GetEnumerator()) {
    $FileName = $Template.Key
    $Description = $Template.Value
    Write-Host "Creating $FileName - $Description" -ForegroundColor Green
    
    $Content = @"
# $($FeatureName.ToUpper()) - $Description

[← Back to Feature Documentation](../README.md)

## Table of Contents
- [Overview](#overview)
- [Details](#details)
- [Related Documentation](#related-documentation)

## Overview
<!-- Brief overview of this aspect of the feature -->

## Details
<!-- Detailed documentation specific to this aspect -->

## Related Documentation

### Core Documentation
- [Model Training Architecture](../../architecture/model_training.md)
- [Data Pipeline](../../architecture/data_pipeline.md)
- [MLflow Guide](../../guides/mlflow.md)

### Supporting Guides
- [Environment Setup](../../guides/environment.md)
- [Code Quality](../../guides/code_quality.md)

### Project Management
- [Changelog](../../CHANGELOG.md)

### API Documentation
- [API Documentation](../../guides/api.md)
- [Testing Guide](../../guides/testing.md)

---
[🔝 Back to Top](#$($FeatureName.ToLower())-$($Description.ToLower() -replace ' ','-'))

Generated: $Timestamp
"@
    
    Set-Content -Path (Join-Path $DocsDir $FileName) -Value $Content -Encoding UTF8
}

# Create example files
$ExampleContent = @"
"""
Example usage of the $FeatureName feature.

This file demonstrates how to:
1. Initialize the feature
2. Configure required parameters
3. Use main functionality
4. Handle common scenarios
"""

def example_usage():
    """Demonstrate basic usage of the feature."""
    pass

def advanced_example():
    """Show advanced use cases and configurations."""
    pass

if __name__ == "__main__":
    example_usage()
    advanced_example()
"@

Set-Content -Path (Join-Path $DocsDir "examples\usage.py") -Value $ExampleContent -Encoding UTF8

# Create test template
$TestFileName = "test_$($FeatureName -replace '-','_').py"
$TestContent = @"
"""
Test suite for the $FeatureName feature.
"""

import pytest
from unittest.mock import Mock, patch

def test_basic_functionality():
    """Test basic feature operations."""
    pass

def test_edge_cases():
    """Test edge cases and error handling."""
    pass

def test_integration():
    """Test integration with other components."""
    pass
"@

Set-Content -Path (Join-Path $DocsDir "tests\$TestFileName") -Value $TestContent -Encoding UTF8

# Create .gitkeep for assets directory
New-Item -ItemType File -Force -Path (Join-Path $DocsDir "assets\.gitkeep") | Out-Null

# Update main README with feature reference
$MainReadme = Join-Path $PSScriptRoot "..\..\README.md"
if (Test-Path $MainReadme) {
    $ReadmeContent = Get-Content $MainReadme -Raw
    if ($ReadmeContent -notmatch "## $FeatureName") {
        Write-Host "Adding feature reference to main README.md" -ForegroundColor Green
        Add-Content -Path $MainReadme -Value "`n## $($FeatureName.ToUpper())`n- [Documentation](features/$FeatureName/README.md)`n"
    }
}

Write-Host "✅ Feature documentation structure created successfully at $DocsDir" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Update feature overview in README.md"
Write-Host "2. Add architecture diagrams in architecture.md"
Write-Host "3. Document components in components.md"
Write-Host "4. Specify API endpoints in api.md"
Write-Host "5. Add test cases in testing.md"
Write-Host "6. Configure monitoring in monitoring.md"
Write-Host "7. Document deployment process in deployment.md"
Write-Host "8. Add usage examples in examples/usage.py"
Write-Host "9. Implement tests in tests\$TestFileName"