# {Feature Name} Architecture

## System Design

### Overview

- High-level architecture diagram
- Key components
- Data flow patterns
- Integration points

### Component Relationships

```mermaid
graph LR
A[Training Data] --> B[MLflow]
B --> C[(Model Registry)]
C --> D[Prediction Service]
D --> E[Monitoring Dashboard]
```

### Data Flow

1. **Input Handling**:
   - Type conversion guards (numeric fields)
   ```python:utils/advanced_goal_features.py
   startLine: 20
   endLine: 35
   ```

2. **Processing**:
   - Two-phase ensemble prediction:
   ```python:models/xgboost_ensemble_model.py
   startLine: 150
   endLine: 180
   ```

3. **Output Generation**:
   - Dual threshold validation (0.3-0.65 range)
   - Probability scores with model agreement tracking

4. **Error Handling**:
   - Column presence validation
   - Temp directory write checks
   ```python:models/xgboost_ensemble_model.py
   startLine: 50
   endLine: 75
   ```

## Technical Decisions

### Technology Choices

**Core ML Stack:**
| Component       | Choice                | Rationale                          |
|-----------------|-----------------------|-------------------------------------|
| ML Framework    | XGBoost/CatBoost      | Handle imbalanced data effectively |
| Tracking        | MLflow                | Model versioning & reproducibility|
| Tuning         | Optuna                | Efficient hyperparameter search    |
| Feature Engine | Custom Pandas-based   | League-specific adaptations        |

**Key Configurations:**
- CPU-only training enforced via XGBoost `tree_method=hist`
- Early stopping rounds: 300-500 depending on dataset size
- Column subsampling range: 0.6-1.0

### Design Patterns

1. **Two-Stage Ensemble**
   - **Use Case**: Precision-recall balance
   - **Implementation**:
   ```python:models/xgboost_ensemble_model.py
   startLine: 250
   endLine: 290
   ```

2. **Model Factory**
   - **Use Case**: Dynamic ensemble construction
   - **Implementation**: UUID-based model selection

3. **Feature Selector**
   - **Use Case**: Dimensionality reduction
   - **Implementation**: XGBoost multi-metric importance
   ```python:utils/feature_selector_api.py
   startLine: 25
   endLine: 45
   ```

### Performance Considerations

**Optimization Techniques:**
- Batch prediction mode for >1000 fixtures
- Column pruning pre-prediction

**Resource Management:**
- Temp directory cleanup after MLflow logging
- Model serialization with protocol=4
- CPU pinning for Optuna trials