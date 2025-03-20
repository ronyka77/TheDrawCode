"""
Stacked model training for soccer draw predictions.
Includes feature engineering with AutoFeat, model optimization with NGBoost and BalancedRandomForest, and MLflow integration.
"""
import os
import sys
import mlflow
import pandas as pd
from pathlib import Path
from autofeat import AutoFeatClassifier
from ngboost import NGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, classification_report
from causalnex.structure import StructureModel
from causalnex.discretiser import Discretiser
from causalnex.network import BayesianNetwork


# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
except Exception as e:
    print(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd().parent)

from utils.create_evaluation_set import import_feature_select_draws_api, setup_mlflow_tracking

# Set up MLflow tracking
experiment_name = "stacked_model_soccer_draw"
mlruns_dir = setup_mlflow_tracking(experiment_name)

def feature_engineering_with_autofeat(X_train, y_train, X_test):
    """Generate interaction terms using AutoFeat with index preservation"""
    # Preserve original indices
    original_indices = X_train.index
    
    # Select top 70 features to reduce memory usage
    top_features = X_train.columns[:70]
    X_train = X_train[top_features]
    X_test = X_test[top_features]

    # Initialize AutoFeat with memory constraints
    af = AutoFeatClassifier(
        feateng_steps=1,
        featsel_runs=2,
        max_gb=16,
        n_jobs=-1,
        verbose=1
    )

    try:
        # Transform features while preserving indices
        X_train_new = af.fit_transform(X_train, y_train)
        
        # Align target using preserved indices
        y_train_new = y_train.loc[original_indices]
        
        # Transform test set
        X_test_new = af.transform(X_test)
        
        return X_train_new, y_train_new, X_test_new
    except Exception as e:
        logger.error(f"AutoFeat transformation failed: {str(e)}")
        raise

def train_model(X_train, y_train, model_type="ngboost", n_jobs=-1, batch_size=10000):
    """Optimized model training with resource utilization"""
    try:
        # Initialize model with optimized parameters
        if model_type == "ngboost":
            model = NGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                minibatch_frac=0.2,
                verbose=1,
                n_jobs=n_jobs,
                early_stopping_rounds=50
            )
        elif model_type == "balanced_forest":
            model = BalancedRandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                n_jobs=n_jobs,
                verbose=1,
                sampling_strategy='auto'
            )
        else:
            raise ValueError("Invalid model_type. Use 'ngboost' or 'balanced_forest'.")
        
        # Batch training for large datasets
        if len(X_train) > batch_size:
            logger.info(f"Training in batches of size {batch_size}")
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                model.fit(X_batch, y_batch)
        else:
            model.fit(X_train, y_train)
            
        return model
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

def optimize_threshold(model, X_test, y_test, min_precision=0.5):
    """Optimize the decision threshold for precision with type validation"""
    try:
        # Ensure y_test is integer type
        y_test = y_test.astype(int)
        
        # Get predicted probabilities
        probas = model.predict_proba(X_test)[:, 1]
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, probas)
        
        # Find optimal threshold meeting minimum precision
        if len(thresholds[precision >= min_precision]) > 0:
            optimal_threshold = thresholds[precision >= min_precision][0]
        else:
            logger.warning("No threshold meets minimum precision. Using default 0.5")
            optimal_threshold = 0.5
            
        return optimal_threshold
    except Exception as e:
        logger.error(f"Threshold optimization failed: {str(e)}")
        raise

def causal_analysis(X_train, y_train, X_test):
    """Prune non-causal features using CausalNex."""
    sm = StructureModel()
    sm.add_edges_from([("referee_draw_rate", "is_draw"), ("defensive_stability", "is_draw")])
    
    discretiser = Discretiser(method="fixed", numeric_split_points=[0.5])
    X_train_discretized = discretiser.fit_transform(X_train)
    
    bn = BayesianNetwork(sm)
    bn.fit(X_train_discretized, y_train)
    
    non_causal_features = [node for node in sm.nodes if sm.in_degree(node) == 0]
    X_train_pruned = X_train.drop(columns=non_causal_features)
    X_test_pruned = X_test.drop(columns=non_causal_features)
    
    return X_train_pruned, X_test_pruned

def main():
    # Load dataset
    X_train, y_train, X_test, y_test = import_feature_select_draws_api()
    
    # Drop non-numeric columns
    non_numeric_cols = ['Referee', 'Venue', 'Home', 'Away']
    X_train = X_train.drop(columns=non_numeric_cols, errors='ignore')
    X_test = X_test.drop(columns=non_numeric_cols, errors='ignore')
    
    # Drop columns with all NaN values
    X_train = X_train.dropna(axis=1, how='all')
    X_test = X_test[X_train.columns]
    
    # Fill remaining NaN values with 0
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    with mlflow.start_run(run_name="stacked_model_soccer_draw"):
        # Step 1: Feature Engineering with AutoFeat
        X_train_new, y_train_new, X_test_new = feature_engineering_with_autofeat(X_train, y_train, X_test)
        mlflow.log_param("autofeat_features", X_train_new.shape[1])
        
        # Step 2: Train NGBoost or BalancedRandomForest
        model = train_model(X_train_new, y_train_new, model_type="ngboost")
        mlflow.log_param("model_type", "NGBoost")
        
        # Step 3: Optimize Threshold
        optimal_threshold = optimize_threshold(model, X_test_new, y_test, min_precision=0.5)
        mlflow.log_param("optimal_threshold", optimal_threshold)
        
        # Step 4: Causal Analysis
        X_train_pruned, X_test_pruned = causal_analysis(X_train_new, y_train_new, X_test_new)
        mlflow.log_param("pruned_features", X_train_pruned.shape[1])
        
        # Retrain model with pruned features
        model.fit(X_train_pruned, y_train_new)
        
        # Evaluate final model
        predictions = (model.predict_proba(X_test_pruned)[:, 1] >= optimal_threshold).astype(int)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        mlflow.log_metrics({"precision": precision, "recall": recall})
        
        # Log classification report
        report = classification_report(y_test, predictions, output_dict=True)
        mlflow.log_dict(report, "classification_report.json")
        
        # Save the final model
        mlflow.pyfunc.log_model("stacked_model", python_model=model)
        
        print(f"Precision: {precision:.2%}, Recall: {recall:.2%}")
        print("Model and metrics logged to MLflow.")

if __name__ == "__main__":
    main() 