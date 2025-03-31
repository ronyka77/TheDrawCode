from pydantic import BaseModel, Field
from typing import List # Import List if needed for future params

class TrainingConfig(BaseModel):
    """Configuration for the ensemble model training."""
    extra_base_model_type: str = Field(
        default='random_forest',
        description="Type of the extra base model to include ('random_forest', 'svm', 'mlp', 'catboost')"
    )
    meta_learner_type: str = Field(
        default='lgb',
        description="Type of the meta-learner ('lgb', 'xgb', 'logistic', 'mlp')"
    )
    calibrate: bool = Field(
        default=False,
        description="Whether to calibrate base model probabilities"
    )
    dynamic_weighting: bool = Field(
        default=True,
        description="Whether to use dynamic weighting for base models"
    )
    target_precision: float = Field(
        default=0.50,
        ge=0,
        le=1,
        description="Target precision for threshold optimization"
    )
    required_recall: float = Field(
        default=0.25,
        ge=0,
        le=1,
        description="Minimum required recall for threshold optimization"
    )
    # Add other relevant parameters from run_ensemble if needed
    # e.g., seasons: List[str] = Field(default_factory=list)

    class Config:
        # Example for generating schema documentation
        schema_extra = {
            "example": {
                "extra_base_model_type": "random_forest",
                "meta_learner_type": "lgb",
                "calibrate": False,
                "dynamic_weighting": True,
                "target_precision": 0.55,
                "required_recall": 0.30
            }
        } 