# Soccer Prediction Project Workflow Orchestrator
# Author: Development Team
# Version: 1.0.0
# Description: Master script to orchestrate feature implementation and bug fixing workflows

# Helper function to validate Git is available
function Test-GitAvailable {
    try {
        git --version | Out-Null
        return $true
    } catch {
        Write-Host "Error: Git is not available in PATH. Please install Git for Windows." -ForegroundColor Red
        return $false
    }
}

# Helper function to validate Python environment
function Test-PythonEnvironment {
    try {
        python -c "import pytest, mlflow, xgboost" 2>$null
        return $true
    } catch {
        Write-Host "Error: Required Python packages not found. Please ensure your environment is activated." -ForegroundColor Red
        Write-Host "Required: pytest, mlflow, xgboost" -ForegroundColor Yellow
        return $false
    }
}

function Show-Menu {
    Clear-Host
    Write-Host "Soccer Prediction Project Workflow`n" -ForegroundColor Yellow
    Write-Host "1) Start New Feature Implementation" -ForegroundColor White
    Write-Host "2) Start Bug Fix" -ForegroundColor White
    Write-Host "3) Run Documentation Checks" -ForegroundColor White
    Write-Host "4) Exit" -ForegroundColor White
    Write-Host "`n"
    
    $choice = Read-Host "Enter your choice (1-4)"
    return $choice
}

function Start-FeatureImplementation {
    Write-Host "Starting Feature Implementation Workflow`n" -ForegroundColor Green

    # Validate environment
    if (-not (Test-GitAvailable) -or -not (Test-PythonEnvironment)) {
        return
    }

    # Step 1: Initial Planning & Requirements
    Write-Host "Step 1: Initial Planning & Requirements" -ForegroundColor Yellow
    
    # Get feature details
    $featureName = Read-Host "Enter feature name (e.g., draw-prediction)"
    if (-not $featureName) {
        Write-Host "Error: Feature name cannot be empty" -ForegroundColor Red
        return
    }
    $featureId = Read-Host "Enter feature ID (e.g., PLAN-42)"
    if (-not $featureId) {
        Write-Host "Error: Feature ID cannot be empty" -ForegroundColor Red
        return
    }
    
    # Create feature branch: only include feature ID (e.g., feature/PLAN-42)
    $branchName = "feature/$featureId"
    Write-Host "Creating branch: $branchName" -ForegroundColor Cyan
    git checkout -b $branchName
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to create feature branch" -ForegroundColor Red
        return
    }
    
    # Step 2: Documentation Setup
    Write-Host "`nStep 2: Setting up Documentation" -ForegroundColor Yellow
    
    # Create feature documentation
    Write-Host "Creating feature documentation..." -ForegroundColor Cyan
    & "$PSScriptRoot\..\templates\feature\Create-FeatureDocs.ps1" -FeatureName $featureName
    
    # Update README
    Write-Host "Updating README..." -ForegroundColor Cyan
    & "$PSScriptRoot\update-readme.ps1"
    
    # Update changelog
    Write-Host "Updating documentation changelog..." -ForegroundColor Cyan
    & "$PSScriptRoot\update-docs-changelog.ps1"
    
    # Step 3: Implementation Phase
    Write-Host "`nStep 3: Implementation Phase" -ForegroundColor Yellow
    Write-Host "Development guidelines:" -ForegroundColor Cyan
    Write-Host "- Follow CPU-only guidelines (tree_method='hist' for XGBoost)" -ForegroundColor White
    Write-Host "- Use ExperimentLogger from logger.py" -ForegroundColor White
    Write-Host "- Validate minimum 1000 samples per training set" -ForegroundColor White
    Write-Host "- Implement early stopping (300-500 rounds)" -ForegroundColor White
    Write-Host "- Clean up temporary files after MLflow logging" -ForegroundColor White
    Write-Host "- Use protocol=4 for model serialization" -ForegroundColor White
    
    # Wait for user to complete implementation
    Read-Host "Press Enter when implementation is complete"
    
    # Step 4: Testing & Verification
    Write-Host "`nStep 4: Testing & Verification" -ForegroundColor Yellow
    
    # Run tests
    Write-Host "Running tests..." -ForegroundColor Cyan
    python -m pytest python_tests/
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Some tests failed. Please review and fix." -ForegroundColor Yellow
        Read-Host "Press Enter when ready to continue"
    }
    
    # Check documentation links
    Write-Host "Checking documentation links..." -ForegroundColor Cyan
    & "$PSScriptRoot\check-doc-links.ps1"
    
    # Step 5: Final Documentation & Commit
    Write-Host "`nStep 5: Final Documentation & Commit" -ForegroundColor Yellow
    Write-Host "Preparing commit..." -ForegroundColor Cyan
    
    # Stage changes
    git add .
    
    # Show changes to be committed
    Write-Host "`nChanges to be committed:" -ForegroundColor Yellow
    git status --short
    
    # Confirm commit
    $confirm = Read-Host "Do you want to commit these changes? (y/n)"
    if ($confirm -eq 'y') {
        $commitMsg = "$featureId`: Add feature '$featureName'"
        git commit -m $commitMsg
        Write-Host "Changes committed successfully!" -ForegroundColor Green
    } else {
        Write-Host "Commit cancelled. You can commit manually when ready." -ForegroundColor Yellow
    }
    
    Write-Host "Feature implementation workflow complete!" -ForegroundColor Green
}

function Start-BugFix {
    Write-Host "Starting Bug Fix Workflow`n" -ForegroundColor Green
    
    # Validate environment
    if (-not (Test-GitAvailable) -or -not (Test-PythonEnvironment)) {
        return
    }
    
    # Step 1: Bug Identification
    Write-Host "Step 1: Bug Identification" -ForegroundColor Yellow
    $bugId = Read-Host "Enter bug ID (e.g., BUG-17)"
    if (-not $bugId) {
        Write-Host "Error: Bug ID cannot be empty" -ForegroundColor Red
        return
    }
    $bugDesc = Read-Host "Enter brief bug description"
    if (-not $bugDesc) {
        Write-Host "Error: Bug description cannot be empty" -ForegroundColor Red
        return
    }
    
    # Create bug fix branch: only include bug ID (e.g., bugfix/BUG-17)
    $branchName = "bugfix/$bugId"
    Write-Host "Creating branch: $branchName" -ForegroundColor Cyan
    git checkout -b $branchName
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to create bugfix branch" -ForegroundColor Red
        return
    }
    
    # Step 2: Task Analysis
    Write-Host "`nStep 2: Task Analysis" -ForegroundColor Yellow
    
    # Step 3: Implementation
    Write-Host "`nStep 3: Implementation" -ForegroundColor Yellow
    Write-Host "Development guidelines:" -ForegroundColor Cyan
    Write-Host "- Make surgical fixes only" -ForegroundColor White
    Write-Host "- Update tests to cover the bug scenario" -ForegroundColor White
    Write-Host "- Follow CPU-only guidelines" -ForegroundColor White
    Write-Host "- Add error logging where appropriate" -ForegroundColor White
    Write-Host "- Document any configuration changes" -ForegroundColor White
    
    # Wait for user to complete fix
    Read-Host "Press Enter when fix is complete"
    
    # Step 4: Testing
    Write-Host "`nStep 4: Testing" -ForegroundColor Yellow
    
    # Run tests
    Write-Host "Running tests..." -ForegroundColor Cyan
    python -m pytest python_tests/
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Some tests failed. Please review and fix." -ForegroundColor Yellow
        Read-Host "Press Enter when ready to continue"
    }
    
    # Check documentation
    Write-Host "Checking documentation..." -ForegroundColor Cyan
    & "$PSScriptRoot\check-doc-links.ps1"
    
    # Step 5: Documentation & Commit
    Write-Host "`nStep 5: Documentation & Commit" -ForegroundColor Yellow
    
    # Update changelog
    Write-Host "Updating documentation changelog..." -ForegroundColor Cyan
    & "$PSScriptRoot\update-docs-changelog.ps1"
    
    # Stage changes
    git add .
    
    # Show changes to be committed
    Write-Host "`nChanges to be committed:" -ForegroundColor Yellow
    git status --short
    
    # Confirm commit
    $confirm = Read-Host "Do you want to commit these changes? (y/n)"
    if ($confirm -eq 'y') {
        $commitMsg = "$bugId`: Fix $bugDesc"
        git commit -m $commitMsg
        Write-Host "Changes committed successfully!" -ForegroundColor Green
    } else {
        Write-Host "Commit cancelled. You can commit manually when ready." -ForegroundColor Yellow
    }
    
    Write-Host "Bug fix workflow complete!" -ForegroundColor Green
}

function Start-DocumentationChecks {
    Write-Host "Running Documentation Checks`n" -ForegroundColor Green
    
    # Check documentation links
    Write-Host "Checking documentation links..." -ForegroundColor Yellow
    & "$PSScriptRoot\check-doc-links.ps1"
    
    # Validate documentation structure
    Write-Host "`nValidating documentation structure..." -ForegroundColor Yellow
    $requiredFiles = @(
        "..\README.md",
        "..\CHANGELOG.md",
        "..\DOCS-CHANGELOG.md",
        "..\architecture\model_training.md",
        "..\guides\mlflow.md"
    )
    
    $allFound = $true
    foreach ($file in $requiredFiles) {
        $fullPath = Join-Path $PSScriptRoot $file
        if (Test-Path $fullPath) {
            Write-Host "✓ Found $file" -ForegroundColor Green
        } else {
            Write-Host "✗ Missing $file" -ForegroundColor Red
            $allFound = $false
        }
    }
    
    # Check YAML frontmatter in documentation files
    Write-Host "`nChecking YAML frontmatter..." -ForegroundColor Yellow
    $mdFiles = Get-ChildItem -Path (Join-Path $PSScriptRoot "..") -Filter "*.md" -Recurse
    foreach ($file in $mdFiles) {
        $content = Get-Content $file.FullName -Raw
        if ($content -match "^---\s*\n.*?\n---") {
            Write-Host "✓ Valid frontmatter in $($file.Name)" -ForegroundColor Green
        } else {
            Write-Host "✗ Missing or invalid frontmatter in $($file.Name)" -ForegroundColor Red
            $allFound = $false
        }
    }
    
    if ($allFound) {
        Write-Host "`nAll documentation checks passed!" -ForegroundColor Green
    } else {
        Write-Host "`nSome documentation checks failed. Please review and fix the issues." -ForegroundColor Yellow
    }
}

# Main workflow loop
$continue = $true
while ($continue) {
    $choice = Show-Menu
    
    switch ($choice) {
        "1" { 
            Start-FeatureImplementation
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Feature implementation completed successfully!" -ForegroundColor Green
            }
        }
        "2" { 
            Start-BugFix
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Bug fix completed successfully!" -ForegroundColor Green
            }
        }
        "3" { 
            Start-DocumentationChecks
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Documentation checks completed successfully!" -ForegroundColor Green
            }
        }
        "4" { 
            $continue = $false
            break
        }
        default { 
            Write-Host "Invalid choice. Please try again." -ForegroundColor Red
        }
    }
    
    if ($continue) {
        Read-Host "`nPress Enter to return to main menu"
    }
}

Write-Host "Thank you for using the workflow orchestrator!" -ForegroundColor Green