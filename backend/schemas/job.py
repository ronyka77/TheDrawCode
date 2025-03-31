from pydantic import BaseModel, Field
from typing import Optional

class TrainingResponse(BaseModel):
    """Response after initiating a training job."""
    job_id: str = Field(description="Unique identifier for the training job")
    status: str = Field(default="started", description="Initial status of the job")
    message: str = Field(description="Information message about the job start")


class TrainingResults(BaseModel):
    """Schema for the results returned after training completion."""
    job_id: str = Field(description="Identifier of the training job")
    status: str = Field(description="Final status ('completed', 'failed')")
    precision: Optional[float] = Field(None, description="Achieved precision")
    recall: Optional[float] = Field(None, description="Achieved recall")
    f1: Optional[float] = Field(None, description="Achieved F1 score")
    threshold: Optional[float] = Field(None, description="Optimal threshold found")
    error_message: Optional[str] = Field(None, description="Error details if the job failed")
    # Add other relevant metrics or artifacts paths if needed 