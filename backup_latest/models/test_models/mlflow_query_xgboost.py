import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

def get_active_parameters():
    """Get parameters from MLflow runs tagged as active."""
    client = MlflowClient()
    mlflow.set_tracking_uri("./mlruns")
    
    # Get all experiments
    experiment = mlflow.get_experiment_by_name("league_hypertuning")
    if not experiment:
        raise ValueError("No experiment found with name 'league_hypertuning'")
    
    # Query for active runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.active = '1'"
    )
    
    league_params = {}
    
    # Safely convert values with type checking
    def safe_convert(value, convert_type, default=0):
        try:
            if pd.isna(value) or value is None:
                return default
            return convert_type(value)
        except (ValueError, TypeError):
            return default
    
    for idx, run in runs.iterrows():
        # Print all available parameters for debugging
        print(f"\nAll parameters for run {idx}:")
        param_cols = [col for col in run.index if col.startswith('params.')]
        for col in param_cols:
            print(f"{col.replace('params.', '')}: {run[col]}")
        
        league_id = safe_convert(run.get('params.league_id'), int)
        
        if league_id > 0:  # Only process if we have a valid league_id
            params = {
                'learning_rate': safe_convert(run.get('params.learning_rate'), float),
                'max_depth': safe_convert(run.get('params.max_depth'), int),
                'min_child_weight': safe_convert(run.get('params.min_child_weight'), int),
                'gamma': safe_convert(run.get('params.gamma'), float),
                'subsample': safe_convert(run.get('params.subsample'), float),
                'colsample_bytree': safe_convert(run.get('params.colsample_bytree'), float),
                'scale_pos_weight': safe_convert(run.get('params.scale_pos_weight'), float),
                'n_estimators': safe_convert(run.get('params.n_estimators'), int),
                'reg_alpha': safe_convert(run.get('params.reg_alpha'), float),
                'reg_lambda': safe_convert(run.get('params.reg_lambda'), float),
                'draw_rate_train': safe_convert(run.get('params.draw_rate_train'), float),
                'draw_rate_val': safe_convert(run.get('params.draw_rate_val'), float),
                'train_samples': safe_convert(run.get('params.train_samples'), int),
                'val_samples': safe_convert(run.get('params.val_samples'), int)
            }
            league_params[league_id] = params
            
    return league_params

if __name__ == "__main__":
    params = get_active_parameters()
    print("\nProcessed parameters by league:")
    import json
    print(json.dumps(params, indent=4)) 