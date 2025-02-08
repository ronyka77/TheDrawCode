"""
Model Context Protocol Server (MCPS) API implementation.

This module implements a FastAPI-based server that provides endpoints for:
- Model training control (start/stop/resume)
- Real-time training status monitoring
- Model evaluation and prediction
- Context metadata management
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# MLflow imports
import mlflow
from mlflow.tracking import MlflowClient

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root ensemble_model: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent)
    print(f"Current directory ensemble_model: {os.getcwd().parent}")
    
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = "C:/Program Files/Git/bin/git.exe"

# Local imports
from utils.logger import ExperimentLogger
from utils.mlflow_utils import MLFlowManager
from state.context_manager import TrainingContext
from async_task.task_manager import TaskManager
from utils.error_monitor import ErrorMonitor


# Initialize components
logger = ExperimentLogger(
    experiment_name="mcps_api",
    log_dir="logs/mcps_api"
)

# Initialize FastAPI app
app = FastAPI(
    title="Model Context Protocol Server (MCPS)",
    description="API for managing model training, evaluation, and context",
    version="1.0.0",
    debug=True
)

# Add CORS middleware to allow all origins (adjust as necessary)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
mlflow_manager = MLFlowManager()
task_manager = TaskManager()
context_manager = TrainingContext()
error_monitor = ErrorMonitor()

# Pydantic models for request/response validation
class TrainingRequest(BaseModel):
    """Training request model."""
    model_type: str = Field(..., description="Type of model to train (xgboost/catboost)")
    experiment_name: str = Field(..., description="Name of the MLflow experiment")
    hyperparameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional hyperparameters for training"
    )

class TrainingResponse(BaseModel):
    """Training response model."""
    run_id: str = Field(..., description="MLflow run ID")
    status: str = Field(..., description="Current training status")
    start_time: str = Field(..., description="Training start time")

class PredictionRequest(BaseModel):
    """Prediction request model."""
    model_uri: str = Field(..., description="URI of the model to use")
    data: Dict[str, List[float]] = Field(..., description="Input data for prediction")

@app.get("/")
async def read_root():
    return {"message": "MCP server is running successfully!"}

@app.post("/train", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
) -> TrainingResponse:
    """Start a new training run.
    
    Args:
        request: Training configuration
        background_tasks: FastAPI background tasks
    
    Returns:
        TrainingResponse with run details
    """
    try:
        # Set up MLflow experiment
        experiment_id = mlflow_manager.setup_experiment(request.experiment_name)
        
        # Start MLflow run
        with mlflow.start_run(experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            
            # Initialize training context
            context_manager.create_context(
                run_id=run_id,
                model_type=request.model_type,
                status="STARTING"
            )
            
            # Queue training task
            background_tasks.add_task(
                task_manager.run_training,
                run_id=run_id,
                model_type=request.model_type,
                hyperparameters=request.hyperparameters
            )
            
            return TrainingResponse(
                run_id=run_id,
                status="STARTING",
                start_time=datetime.now().isoformat()
            )
            
    except Exception as e:
        error_monitor.log_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{run_id}")
async def get_training_status(run_id: str) -> Dict[str, Any]:
    """Get status of a training run.
    
    Args:
        run_id: MLflow run ID
    
    Returns:
        Dict with current training status and metrics
    """
    try:
        # Get context from manager
        context = context_manager.get_context(run_id)
        if not context:
            raise HTTPException(
                status_code=404,
                detail=f"No training context found for run_id: {run_id}"
            )
            
        # Get MLflow metrics
        client = MlflowClient()
        run = client.get_run(run_id)
        
        return {
            "status": context.status,
            "metrics": run.data.metrics,
            "start_time": context.start_time,
            "last_updated": context.last_updated
        }
        
    except Exception as e:
        error_monitor.log_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop/{run_id}")
async def stop_training(run_id: str) -> Dict[str, str]:
    """Stop a training run.
    
    Args:
        run_id: MLflow run ID
    
    Returns:
        Dict with status message
    """
    try:
        # Stop the training task
        task_manager.stop_training(run_id)
        
        # Update context
        context_manager.update_status(run_id, "STOPPED")
        
        return {"status": "Training stopped successfully"}
        
    except Exception as e:
        error_monitor.log_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest) -> Dict[str, Any]:
    """Make predictions using a trained model.
    
    Args:
        request: Prediction request with model URI and input data
    
    Returns:
        Dict with predictions
    """
    try:
        # Load model from MLflow
        model = mlflow.pyfunc.load_model(request.model_uri)
        
        # Convert input data to pandas DataFrame
        import pandas as pd
        input_df = pd.DataFrame(request.data)
        
        # Make predictions
        predictions = model.predict(input_df)
        
        return {
            "predictions": predictions.tolist(),
            "model_uri": request.model_uri
        }
        
    except Exception as e:
        error_monitor.log_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/experiments")
async def list_experiments() -> List[Dict[str, Any]]:
    """List all MLflow experiments.
    
    Returns:
        List of experiment details
    """
    try:
        client = MlflowClient()
        experiments = client.search_experiments()
        
        return [{
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "artifact_location": exp.artifact_location,
            "lifecycle_stage": exp.lifecycle_stage
        } for exp in experiments]
        
    except Exception as e:
        error_monitor.log_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/runs/{experiment_id}")
async def list_runs(experiment_id: str) -> List[Dict[str, Any]]:
    """List all runs for an experiment.
    
    Args:
        experiment_id: MLflow experiment ID
    
    Returns:
        List of run details
    """
    try:
        client = MlflowClient()
        runs = client.search_runs(experiment_id)
        
        return [{
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "metrics": run.data.metrics
        } for run in runs]
        
    except Exception as e:
        error_monitor.log_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000) 