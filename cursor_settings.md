# Cursor Settings - Rules for AI

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Core Principles](#core-principles)
- [Example Workflow](#example-workflow)
- [Critical Rules](#critical-rules)
- [File Management](#file-management)
- [Change Control](#change-control)

<a name="overview"></a>
## Overview
These settings ensure consistent development practices across the Soccer Prediction Project ecosystem. They enforce:
- Working memory preservation through composer-history tracking
- Compliance with project plans and documentation standards
- Surgical code modifications to maintain system stability
- Reproducible environment configurations via MLflow logging

TheDrawCode/
├── models/                 # ML model implementations
├── predictors/            # Prediction logic
├── utils/                 # Utility functions
├── python_tests/         # Test suite
├── docs/                 # Documentation
│   ├── templates/       # Documentation templates
│   │   └── feature/    # Feature-specific docs
│   └── working-memory/ # Project tracking
├── data/                 # Data storage
├── mlruns/              # MLflow tracking
└── logs/                # Application logs

<a name="quick-start"></a>
## Quick Start
1. Copy-paste these rules into Cursor's 'Rules for AI' settings:
```python
# BEGIN RULES BLOCK
Always greet me with my name - Ricsi.

Review .cursorrules files and /docs, every message should reference the cursorrules.

It is very important to have a working memory.
!!Always check these files for current project state before any work!!:

1. /docs/plan.md - Main project plan and task tracking
2. /docs/plan-podcast.md - Podcast feature specific planning
3. Output plan updates before starting work
4. Reference plan number in all communications

All models should be in models/ and predictors in predictors/

Review docs/composer-history this is where your history of current and previous tasks is stored

Every run should be using composer history and .plan and referencing the .cursorrules file

Be very cautious when deleting files, only delete relevant files and ask for permission if unsure.
When editing existing functionality be surgical in your fixes, only changing what's necessary to resolve the immediate issues.

Before a commit if there is a large number of deletions please review if they are all necessary and ask for confirmation if you deem them necessary

Always update the .plan file.
Always run a command to get the current date and time, do not hallucinate it
# END RULES BLOCK
```

<a name="core-principles"></a>
## Core Principles
- **Working Memory**: Maintain state through [composer-history](/docs/composer-history)
- **Plan Adherence**: Coordinate via [Master Plan](/docs/plan.md) and [Podcast Roadmap](/docs/plan-podcast.md)
- **Surgical Edits**: Preserve existing MLflow tracking and model registry patterns
- **Environment Safety**: Enforce CPU-only training via XGBoost `device=cpu`

<a name="example-workflow"></a>
## Example Workflow
**Task:** Update project plan with new feature
1. Check [composer-history](/docs/composer-history) for recent changes
2. Review active tasks in [plan.md](/docs/plan.md#current-sprint)
3. Execute `date +"%Y-%m-%d %H:%M:%S"` for timestamp
4. Make focused changes referencing .cursorrules
5. Validate with `check_samples.py --min-rows 1000`
6. Commit with plan version tag: `git commit -m "PLAN-42: Update threshold validation"`

# Recommended Documentation Structure

/docs/
├── plan.md                    # Project roadmap and task tracking
├── composer-history/          # Task history and decisions
├── guides/                    # Developer guides
│   ├── setup.md              # Environment setup
│   ├── mlflow.md             # MLflow usage
│   └── troubleshooting.md    # Common issues
└── templates/                 # (existing structure)

<a name="critical-rules"></a>
## Critical Rules
```python
# ALWAYS DO FIRST
!!Always check these files for current project state before any work!!:
1. "/docs/plan.md"  # Contains active sprint goals
2. "/docs/plan-podcast.md"  # Podcast feature requirements
3. ".cursorrules"  # Current project constraints
```

<a name="file-management"></a>
## File Management
| Directory | Purpose | Critical Files |
|-----------|---------|-----------------|
| `models/` | ML Models | [xgboost_api_model.py](/models/xgboost_api_model.py) |
| `predictors/` | Prediction handlers | [ensemble_predictor.py](/predictors/ensemble_predictor.py) |
| `docs/` | Planning | [composer-history](/docs/composer-history) |

<a name="change-control"></a>
## Change Control
```python
# REQUIRED BEFORE COMMITTING
if deletion_count > 3:
    confirm("Are these deletions essential for current PLAN objective?")
    
if modifying models/xgboost_*.py:
    validate_mlflow_logging()
```

> **Summary of Changes v2.1:**
> - Added timestamp validation workflow
> - Linked core principles to project architecture
> - Integrated plan version tagging example
> - Clarified model directory requirements
