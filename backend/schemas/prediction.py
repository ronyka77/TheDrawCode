from pydantic import BaseModel, Field
from typing import Optional, Any

class PredictionInput(BaseModel):
    """Input features for making a prediction."""
    # Define fields based on the features your model expects
    # This is highly dependent on your feature engineering
    # Example:
    feature1: float
    feature2: int
    feature3: Optional[str] = None
    # ... add all required features ...

    class Config:
        schema_extra = {
            "example": {
                "feature1": 10.5,
                "feature2": 5,
                "feature3": "category_A"
                # ... example values ...
            }
        }


class PredictionOutput(BaseModel):
    """Output of a prediction request."""
    prediction: Any = Field(description="The prediction result (e.g., probability, class label)")
    # Optional: Add probability if it's a classification task
    probability: Optional[float] = Field(None, description="Predicted probability (if applicable)") 