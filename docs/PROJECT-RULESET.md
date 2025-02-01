# Soccer Prediction Project Rule Set (v2.1+)

This document summarizes the development and operational guidelines for our Soccer Prediction Project. It consolidates rules from our internal cursorrules, process documents, and architecture guides. Every team member must review these rules before starting work.

---

## 1. General Principles

- **Environment & Hardware**:  
  - **OS**: Windows 11  
  - **Training Mode**: CPU-only (no GPU support); all model training configurations (e.g., using `xgb.tree_method='hist'`) must enforce CPU usage.
- **Surgical Code Changes**:  
  - When fixing bugs or enhancing features, change only what is necessary. When in doubt about deletions or large fixes, request explicit permission.
- **Versioning & Reproducibility**:  
  - Use semantic versioning for all models, documentation, and code changes.
  - Each commit must reference the associated plan number from [docs/plan.md](../plan.md).

---

## 2. Documentation & Communication

- **Documentation Reviews**:  
  - Always consult the latest versions of:
    - [Project Plan](../plan.md)
    - [Podcast Feature Roadmap](../plan-podcast.md)
    - [Development History](../composer-history)
    - [Cursor Settings](../docs/cursor_settings.md)
- **Plan Updates**:  
  - Update the project plan before starting any work.
  - Reference the plan number and update the changelog entries as shown in [docs/DOCS-CHANGELOG.md](../DOCS-CHANGELOG.md).
- **Relative Linking & Code Annotations**:  
  - Use relative paths for links in documentation.
  - Code examples must include the language identifier and file path in the opening code blocks.

---

## 3. Logging & MLflow Integration

- **Standard Logging**:  
  - Use the `ExperimentLogger` defined in [utils/logger.py](../utils/logger.py).  
  - Logs must be structured in JSON, use ISO 8601 timestamps, and rotate at 10MB per file.
  - Log key events: start/end of training, parameter/metric logging, data validation steps, and feature engineering phases.
- **MLflow Guidelines**:  
  - Always register models using a registry name formatted as: `{model_type}_YYYYMMDD_HHMM`.  
  - Log artifacts including feature importance plots and training analysis reports.
  - Follow the procedures in [docs/guides/mlflow.md](../docs/guides/mlflow.md) for tracking and artifact management.

---

## 4. Data Pipeline & Validation

- **Data Quality Assurance**:  
  - Follow the architecture guidelines in [docs/architecture/data_pipeline.md](../docs/architecture/data_pipeline.md).  
  - Validate training data with the utility `validate_training_data()` ensuring a minimum of 1000 samples and proper numeric conversions.
- **Data Validation Rules**:  
  - Enforce required columns, data type correctness, and consistency as detailed in [docs/guides/data_validation_rules.md](../docs/guides/data_validation_rules.md).
  - Any violation (missing columns, incorrect types, etc.) must be logged through the standard error handling system with appropriate error codes (see [docs/error_handling.md](../docs/error_handling.md)).

---

## 5. Model Training & Evaluation

- **Model Training**:  
  - Utilize XGBoost with CPU-optimized configurations (_e.g._ `tree_method='hist'` and early stopping rounds between 300–500).
  - Log all training parameters and metrics via MLflow as described in [docs/architecture/model_training.md](../docs/architecture/model_training.md).
  - In case of convergence issues, apply the learning rate adjustment function (`adjust_learning_rate`) and document any changes.
- **Evaluation & Stability**:  
  - Implement cross-validation and stability checks (see [docs/architecture/model_training.md](../docs/architecture/model_training.md)) to ensure reliable prediction outcomes.
  - Model evaluation metrics should be logged, and any unusual patterns must trigger a review with detailed logging.

---

## 6. Prediction Service & Feature Selection

- **Prediction Workflow**:  
  - Follow the service design in [docs/architecture/prediction.md](../docs/architecture/prediction.md) for data preprocessing, feature selection, and model prediction.
  - Validate features rigorously (e.g., with `validate_features()`) before making predictions.
- **Feature Selection Guidelines**:  
  - Implement the feature selection logic as defined in [utils/feature_selection.py](../utils/feature_selection.py).
  - Use composite scoring (combining gain, weight, and cover metrics) and log feature selection metrics in MLflow (see [docs/guides/feature_selection.md](../docs/guides/feature_selection.md) and [docs/guides/feature_selection_guide.md](../docs/guides/feature_selection_guide.md)).

---

## 7. Error Handling & Recovery

- **Standard Error Codes**:  
  - Use predefined error codes (see `DataProcessingError` in [docs/error_handling.md](../docs/error_handling.md)) for consistent error reporting.
- **Logging Failures**:  
  - All exceptions must be caught and logged using the `ExperimentLogger`.
  - Implement retry mechanisms and escalate persistent issues according to our recovery procedures laid out in [docs/guides/recovery_procedures.md](../docs/guides/recovery_procedures.md).

---

## 8. System Monitoring & Performance

- **Monitoring**:  
  - Follow [docs/guides/system_monitoring.md](../docs/guides/system_monitoring.md) to routinely check error patterns, log sizes, and system resource usage.
  - Regularly review performance metrics and update thresholds as needed.
- **Automated Scripts**:  
  - Use provided PowerShell scripts (e.g., `docs/scripts/workflow.ps1`, `update-docs-changelog.ps1`) to orchestrate testing, documentation link checking, and commit workflows.

---

## 9. Process & Commit Workflow

- **Development Workflow**:  
  - Adhere to our structured development process:
    1. Update plan and documentation.
    2. Implement surgical fixes with targeted testing.
    3. Run the full test suite (`python -m pytest python_tests/`).
    4. Validate documentation links using `docs/scripts/check-doc-links.ps1`.
    5. Commit changes with comprehensive commit messages referencing plan numbers.
- **Final Checks**:  
  - Before final commits, ensure all documentation is updated and consistent with our changelog guidelines in [docs/DOCS-CHANGELOG.md](../docs/DOCS-CHANGELOG.md).

---

## 10. Final Notes

- All team members must abide by the rules defined above and ensure alignment with the project's overarching goals of high-precision draw predictions, robust error handling, and full reproducibility.
- For any deviations or uncertainties, please review the corresponding sections in our documentation (e.g., [docs/plan.md](../docs/plan.md), [docs/composer-history](../docs/composer-history)) and consult with the team lead.

*Generated on: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")* 