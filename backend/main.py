from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, UploadFile, File
import uuid # Import uuid for generating job IDs
import json
import asyncio # Import asyncio for sleep
from typing import Dict, List, Tuple, Optional
import pandas as pd # Import pandas
import io # To read file contents
import os
import sys

try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)
except Exception as e:
    print(f"Error setting project root: {e}")

# Import schemas from the new package structure
from backend.schemas import (
    TrainingConfig,
    TrainingResponse,
    TrainingResults, # Now we'll use this
    PredictionInput,
    PredictionOutput
)

# --- Add imports for prediction logic ---
from predictors.predict_ensemble import DrawPredictor # Assuming this path works
from pydantic import BaseModel

# In-memory storage for configuration (replace with persistent storage later)
current_config = TrainingConfig() # Initialize with default values

# In-memory storage for active WebSocket connections by job_id
active_connections: Dict[str, WebSocket] = {}

# In-memory storage for logs by job_id (simple temporary solution)
job_logs: Dict[str, List[str]] = {}

# In-memory storage for results by job_id (NEW)
job_results: Dict[str, TrainingResults] = {}

app = FastAPI(
    title="Soccer Prediction Backend",
    description="API for managing soccer prediction model training and predictions.",
    version="0.1.0",
)

@app.get("/")
async def read_root():
    """Root endpoint returning a welcome message."""
    return {"message": "Welcome to the Soccer Prediction Backend API!"}

# --- WebSocket Manager ---

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        self.active_connections[job_id] = websocket
        print(f"WebSocket connected for job_id: {job_id}")

    def disconnect(self, job_id: str):
        if job_id in self.active_connections:
            del self.active_connections[job_id]
            print(f"WebSocket disconnected for job_id: {job_id}")

    async def send_log(self, job_id: str, message: str):
        if job_id in self.active_connections:
            await self.active_connections[job_id].send_text(
                json.dumps({"type": "log", "data": message})
            )
            # Also store in memory for retrieval on reconnect
            if job_id not in job_logs:
                job_logs[job_id] = []
            job_logs[job_id].append(message)
            print(f"Log sent to job_id {job_id}: {message}")

    async def send_status(self, job_id: str, status: str):
        if job_id in self.active_connections:
            await self.active_connections[job_id].send_text(
                json.dumps({"type": "status", "data": status})
            )
            print(f"Status update sent to job_id {job_id}: {status}")

manager = ConnectionManager()

# --- WebSocket Endpoint ---
@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time communication.
    Each client connects with a job_id to receive updates for a specific job.
    """
    await manager.connect(websocket, job_id)
    # Send any existing logs for this job (if reconnecting)
    if job_id in job_logs:
        for log in job_logs[job_id]:
            await manager.send_log(job_id, log)
    try:
        # Simple ping/pong mechanism to keep connection alive
        while True:
            # Wait for message from client (could be used for more complex interaction)
            data = await websocket.receive_text()
            # Echo back the message for now
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        manager.disconnect(job_id)

# --- Configuration Endpoints ---

@app.get("/config", response_model=TrainingConfig)
async def get_configuration():
    """Retrieve the current training configuration."""
    return current_config

@app.put("/config", response_model=TrainingConfig)
async def update_configuration(config: TrainingConfig):
    """Update the training configuration."""
    global current_config
    current_config = config
    print(f"Configuration updated: {current_config.dict()}") # Log to console for now
    return current_config

# --- Background Training Task (Modified to store results) ---
async def run_training_background(job_id: str, config: TrainingConfig, manager: ConnectionManager):
    """Placeholder function to simulate training and send logs/status via WebSocket."""
    print(f"Background task started for job_id: {job_id}")
    await manager.send_status(job_id, "initializing")
    await manager.send_log(job_id, "Training process started.")
    await manager.send_log(job_id, f"Configuration: {config.dict()}")
    await asyncio.sleep(2) # Simulate setup

    try:
        await manager.send_status(job_id, "loading_data")
        await manager.send_log(job_id, "Loading training data...")
        await asyncio.sleep(3) # Simulate data loading
        await manager.send_log(job_id, "Data loading complete.")
        await manager.send_status(job_id, "training")
        await manager.send_log(job_id, "Starting model training loop...")
        # Simulate training progress
        for i in range(1, 6):
            await asyncio.sleep(2) # Simulate work for one iteration
            await manager.send_log(job_id, f"Training iteration {i*10}/50 complete.")
        
        await manager.send_log(job_id, "Training finished.")
        await manager.send_status(job_id, "evaluating")
        await manager.send_log(job_id, "Evaluating model...")
        await asyncio.sleep(2) # Simulate evaluation
        # Store dummy results in memory (NEW)
        dummy_results = TrainingResults(
            precision=0.75, 
            recall=0.65,
            f1_score=0.70,
            optimal_threshold=0.55,
            feature_importance=[
                {"feature": "feature_A", "importance": 0.3},
                {"feature": "feature_B", "importance": 0.25},
                {"feature": "feature_C", "importance": 0.15}
            ]
        )
        job_results[job_id] = dummy_results
        await manager.send_log(job_id, "Results stored.")
        # --- End NEW section ---
        await manager.send_log(job_id, "Training and evaluation completed successfully.")
        await manager.send_status(job_id, "completed") # Send final status
        print(f"Background task finished successfully for job_id: {job_id}")
    except Exception as e:
        error_message = f"Training failed for job {job_id}: {str(e)}"
        print(error_message)
        await manager.send_log(job_id, error_message)
        await manager.send_status(job_id, "failed") # Send final error status
    finally:
        # Clean up logs for this job? Optional, depends on requirements.
        # if job_id in job_logs:
        #     del job_logs[job_id]
        pass # No specific cleanup for now

# --- Training Endpoint (Updated) ---

@app.post("/train", response_model=TrainingResponse)
async def start_training_job(background_tasks: BackgroundTasks, config: TrainingConfig = None):
    """
    Initiate a new training job using background tasks.
    Sends logs and status updates via WebSocket.
    """
    config_to_use = config if config else current_config
    job_id = str(uuid.uuid4()) # Generate a unique ID for the job
    print(f"Training job {job_id} requested with config: {config_to_use.dict()}")
    
    # Initialize log storage for this job (important for reconnects)
    job_logs[job_id] = []
    
    # Add the actual training logic to run in the background
    background_tasks.add_task(
        run_training_background, 
        job_id, 
        config_to_use, 
        manager # Pass the connection manager instance
    )
    
    # Return immediately, confirming the job submission
    return TrainingResponse(
        job_id=job_id,
        status="submitted",
        message=f"Training job {job_id} submitted successfully."
    )

# --- Results Endpoint (NEW) ---

@app.get("/results/{job_id}", response_model=TrainingResults)
async def get_training_results(job_id: str):
    """Retrieve the results for a completed training job."""
    if job_id not in job_results:
        raise HTTPException(status_code=404, detail=f"Results not found for job_id: {job_id}. Job may not exist, be running, or failed.")
    return job_results[job_id]

# --- Prediction Endpoint (Modified for File Upload & Real Prediction) ---

# Define a more detailed response structure
class PredictionResultItem(BaseModel):
    # Include identifier if possible, e.g., fixture_id or row index
    # For now, just index
    index: int 
    prediction: int # 0 or 1
    probability: float

class DetailedPredictionResponse(BaseModel):
    message: str
    file_name: str
    data_shape: Optional[Tuple[int, int]] = None
    num_predictions_processed: int = 0
    num_positive_predictions: int = 0
    prediction_rate: float = 0.0
    predictions_sample: List[PredictionResultItem] = [] # Return a sample

# Define the model URI - TODO: Make this configurable or load from registry
MODEL_RUN_ID = '94b35c3a239f493880d6f35e4fde7035' 
PREDICTION_MODEL_URI = f"runs:/{MODEL_RUN_ID}/ensemble_model"
MAX_PREDICTIONS_IN_RESPONSE = 100 # Limit response size

@app.post("/predict", response_model=DetailedPredictionResponse) # Use new response model
async def make_prediction_from_file(file: UploadFile = File(...)):
    """
    Make predictions using the DrawPredictor model based on input features 
    from an uploaded XLSX file.
    """
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an XLSX file.")
    print(f"Prediction requested with file: {file.filename}")
    try:
        # Read the uploaded file content into a pandas DataFrame
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        print(f"Successfully loaded data from {file.filename}. Shape: {df.shape}")

        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded XLSX file is empty or has no data.")
        # --- Use DrawPredictor for prediction --- 
        print(f"Loading model from URI: {PREDICTION_MODEL_URI}")
        try:
            predictor = DrawPredictor(model_uri=PREDICTION_MODEL_URI)
        except Exception as model_load_error:
            print(f"Failed to load prediction model: {model_load_error}")
            raise HTTPException(status_code=500, detail=f"Failed to load prediction model: {str(model_load_error)}")
        print("Model loaded. Preparing data for prediction...")
        # Ensure data types are compatible if needed (DrawPredictor might handle this)
        numeric_columns = df.select_dtypes(include=['number']).columns
        df = df.astype({col: 'float64' for col in numeric_columns})

        # Make predictions using the predictor class
        # The predict method should handle validation (_validate_input)
        prediction_results_dict = predictor.predict(df)
        print(f"Prediction completed. Processing {prediction_results_dict.get('num_predictions', 0)} predictions.")
        # Prepare response sample
        predictions_sample = []
        pred_list = prediction_results_dict.get('predictions', [])
        prob_list = prediction_results_dict.get('draw_probabilities', [])
        num_to_sample = min(len(pred_list), MAX_PREDICTIONS_IN_RESPONSE)
        
        for i in range(num_to_sample):
            predictions_sample.append(
                PredictionResultItem(
                    index=i, # Use actual index or identifier if available in df
                    prediction=int(pred_list[i]),
                    probability=float(prob_list[i])
                )
            )
        # --- End Prediction Logic --- 
        return DetailedPredictionResponse(
            message="Predictions processed successfully.",
            file_name=file.filename,
            data_shape=df.shape,
            num_predictions_processed=prediction_results_dict.get('num_predictions', 0),
            num_positive_predictions=prediction_results_dict.get('positive_predictions', 0),
            prediction_rate=prediction_results_dict.get('prediction_rate', 0.0),
            predictions_sample=predictions_sample
        )
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded XLSX file is empty.")
    except ValueError as ve:
        # Catch validation errors from DrawPredictor (_validate_input)
        print(f"Input validation error: {ve}")
        raise HTTPException(status_code=400, detail=f"Input validation error: {str(ve)}")
    except Exception as e:
        print(f"Error processing prediction file {file.filename}: {e}")
        # Raise a generic server error, consider more specific errors
        raise HTTPException(status_code=500, detail=f"Failed to process prediction file: {str(e)}")

# Add more endpoints later for configuration, training, prediction, websockets, etc.

# To run the app: uvicorn backend.main:app --reload --port 8000 