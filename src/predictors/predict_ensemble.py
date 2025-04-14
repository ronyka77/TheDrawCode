import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

# Disable pandas SettingWithCopyWarning
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Suppress pandas chained assignment warnings
pd.options.mode.chained_assignment = None  # default='warn'

from src.utils.create_evaluation_set import (
    create_prediction_set_ensemble,
    get_real_api_scores_from_excel,
    setup_mlflow_tracking,
)

experiment_name = "ensemble_model_improved"
mlruns_dir = setup_mlflow_tracking(experiment_name)
# Set fixed seed and hash seed for determinism
SEED = 19
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Restrict parallel threads across various libraries
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"


class DrawPredictor:
    """Predictor class for draw predictions using the stacked model."""

    def __init__(self, model_uri: str):
        """Initialize predictor with model URI."""
        # Set up MLflow tracking URI based on current environment
        os.getcwd()
        try:
            self.model = mlflow.sklearn.load_model(model_uri)
            self.test_model = mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = mlflow.pyfunc.load_model(model_uri)
            self.test_model = self.model
        try:
            # Retrieve the optimal threshold if set during training.
            if hasattr(self.model, "optimal_threshold"):
                self.threshold = self.model.optimal_threshold
                print(f"Using model's optimal threshold: {self.threshold:.2%}")
            # elif hasattr(self.test_model, "metadata") and "threshold" in self.test_model.metadata.get_all_tags():
            #     self.threshold = float(self.test_model.metadata.get_tag("threshold"))
            #     print(f"Using model's threshold: {self.threshold:.2%}")
            else:
                self.threshold = 0.27
                print("No optimal threshold found in model, using default 27% threshold")
            # Get feature names from signature if available.
            if self.test_model.metadata.signature:
                self.required_features = self.test_model.metadata.signature.inputs.input_names()
                print(f"Features from signature: {len(self.required_features)}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.threshold = 0.27

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input dataframe has all required columns."""
        missing_cols = set(self.required_features) - set(df.columns)
        # Drop columns not in required features
        extra_cols = set(df.columns) - set(self.required_features)
        if extra_cols:
            df.drop(columns=list(extra_cols), inplace=True, errors="ignore")
            print(f"Dropped columns: {len(extra_cols)}")
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def predict(self, df: pd.DataFrame) -> dict[str, Any]:
        """Make predictions and return results with probabilities."""
        # Validate input if needed.
        self._validate_input(df)

        # Get probabilities - our ensemble model returns a 1D array of positive class probabilities
        try:
            predictions = self.model.predict(df)
            pos_probas = self.model.predict_proba(df)
            # Ensure we have a 1D numpy array
            if not isinstance(pos_probas, np.ndarray):
                pos_probas = np.array(pos_probas)
        except Exception as e:
            print(f"Error predicting: {e}")
            if "use_label_encoder" in str(e):
                print("Attribute error due to missing 'use_label_encoder'. Patching model...")
                self.model.use_label_encoder = False
                predictions = self.model.predict(df)
                pos_probas = self.model.predict_proba(df)

        results = {
            "predictions": predictions.tolist(),
            "draw_probabilities": pos_probas.tolist(),
            "threshold_used": self.threshold,
            "num_predictions": len(predictions),
            "positive_predictions": int(np.sum(predictions)),
            "prediction_rate": float(np.mean(predictions)),
        }
        print(f"Prediction rate: {results['prediction_rate']}")
        return results

    def _find_optimal_threshold(
        self, model: XGBClassifier, features_val: pd.DataFrame, target_val: pd.Series
    ) -> tuple[float, dict[str, float]]:
        """Find optimal prediction threshold prioritizing precision while maintaining recall.
        Args:
            model: Trained XGBoost model
            features_val: Validation features
            target_val: Validation targets
        Returns:
            Tuple of (optimal threshold, metrics dictionary)
        """
        try:
            prediction_df = features_val.copy()
            prediction_df = prediction_df[self.required_features]
            probas = self.model.predict_proba(prediction_df)[:, 1]
            best_metrics = {"precision": 0, "recall": 0, "f1": 0, "threshold": 0.5}
            best_score = 0

            # Focus on higher thresholds for better precision, starting from 0.5
            for threshold in np.arange(0.5, 0.65, 0.01):
                preds = (probas >= threshold).astype(int)
                true_positives = ((preds == 1) & (target_val == 1)).sum()
                false_positives = ((preds == 1) & (target_val == 0)).sum()
                ((preds == 0) & (target_val == 0)).sum()
                false_negatives = ((preds == 0) & (target_val == 1)).sum()
                # Calculate metrics
                recall = (
                    true_positives / (true_positives + false_negatives)
                    if (true_positives + false_negatives) > 0
                    else 0
                )
                # Only consider thresholds that meet minimum recall
                if recall >= 0.20:
                    precision = (
                        true_positives / (true_positives + false_positives)
                        if (true_positives + false_positives) > 0
                        else 0
                    )
                    f1 = f1_score(target_val, preds)
                    # Modified scoring to prioritize precision
                    score = precision

                    if score > best_score:
                        best_score = score
                        best_metrics.update(
                            {
                                "precision": precision,
                                "recall": recall,
                                "f1": f1,
                                "threshold": threshold,
                            }
                        )
            self.threshold = best_metrics["threshold"]
            print(f"Optimal threshold set to {self.threshold}")

            if best_metrics["recall"] < 0.20:
                print(
                    f"Could not find threshold meeting recall requirement. "
                    f"Best recall: {best_metrics['recall']:.4f}"
                    f"Best precision: {best_metrics['precision']:.4f}"
                )
            print(
                f"New best threshold {best_metrics['threshold']:.3f}: "
                f"Precision={best_metrics['precision']:.4f}, Recall={best_metrics['recall']:.4f}"
            )
            return self.threshold, best_metrics

        except Exception as e:
            print(f"Error in threshold optimization: {str(e)}")
            raise


def make_prediction(prediction_data, model_uri, real_scores_df) -> pd.DataFrame:
    """Make predictions and return results with probabilities."""
    try:
        # Initialize default values
        precision = 0.0
        draws_recall = 0.0

        # Initialize predictor
        predictor = DrawPredictor(model_uri)
        prediction_df = prediction_data.copy()

        # Ensure data types are compatible with model expectations
        # Convert numeric columns to float64 to match model expectations
        numeric_columns = prediction_df.select_dtypes(include=["number"]).columns
        prediction_df = prediction_df.astype({col: "float64" for col in numeric_columns})

        # Add column validation
        predictor._validate_input(prediction_df)

        # Add dtype consistency check with proper DataFrame handling
        if not isinstance(prediction_data, pd.DataFrame):
            raise TypeError("prediction_data must be a pandas DataFrame")

        prediction_df = prediction_df[predictor.required_features]
        results = predictor.predict(prediction_df)
        print("Prediction successful...")
        # Merge prediction data with real scores to get is_draw column
        predict_df = prediction_df.merge(
            real_scores_df[["fixture_id", "is_draw"]], on="fixture_id", how="left"
        )
        # Drop rows with NaN in is_draw column
        predict_df = predict_df.dropna(subset=["is_draw"])
        print(
            f"Merged prediction data with real scores and dropped NaN is_draw. Shape: {predict_df.shape}"
        )

        # Add predictions to dataframe using .loc to avoid SettingWithCopyWarning
        prediction_df = prediction_df.copy()  # Create explicit copy
        prediction_df.loc[:, "draw_predicted"] = results["predictions"]
        prediction_df.loc[:, "draw_probability"] = [
            round(prob, 2) for prob in results["draw_probabilities"]
        ]
        # Get real scores and merge - this is where the error occurs
        if "fixture_id" in prediction_data.columns:
            print(f"prediction_data.columns: {prediction_data.shape}")
            # valid_fixture_ids = prediction_df['fixture_id'].dropna().astype('Int64').tolist()
            if not real_scores_df.empty:  # Only proceed if we have real scores
                # Ensure is_draw column exists and is properly formatted
                if "is_draw" not in real_scores_df.columns:
                    if "match_outcome" in real_scores_df.columns:
                        # Create is_draw from match_outcome if available
                        real_scores_df["is_draw"] = (real_scores_df["match_outcome"] == 2).astype(
                            int
                        )
                    else:
                        print("Warning: No match outcome data available in real scores")
                        real_scores_df["is_draw"] = None
                # Merge with validation data
                prediction_df = prediction_df.merge(
                    prediction_data[
                        ["fixture_id", "league_name"]
                        + [
                            col
                            for col in prediction_data.columns
                            if col not in prediction_df.columns
                        ]
                    ],
                    on="fixture_id",
                    how="left",
                )
                # Merge with validation data
                matches_with_results = prediction_df.merge(
                    real_scores_df, on="fixture_id", how="left"
                )

                # Set fixture_id as the last column in dataset by reordering columns
                if "fixture_id" in matches_with_results.columns:
                    # Get all columns except fixture_id
                    other_cols = [
                        col for col in matches_with_results.columns if col != "fixture_id"
                    ]
                    # Reorder columns with fixture_id at the end
                    matches_with_results = matches_with_results[other_cols + ["fixture_id"]]
                # Ensure is_draw is properly typed
                if "is_draw" in matches_with_results.columns:
                    matches_with_results["is_draw"] = (
                        matches_with_results["is_draw"].fillna(-1).astype(int)
                    )
                if len(matches_with_results) > 0 and "is_draw" in matches_with_results.columns:
                    # Filter out rows without valid is_draw values
                    valid_matches = matches_with_results[matches_with_results["is_draw"] != -1]

                    if len(valid_matches) > 0:
                        # Calculate metrics using valid matches
                        true_positives = (
                            (valid_matches["draw_predicted"] == 1) & (valid_matches["is_draw"] == 1)
                        ).sum()
                        false_positives = (
                            (valid_matches["draw_predicted"] == 1) & (valid_matches["is_draw"] == 0)
                        ).sum()
                        true_negatives = (
                            (valid_matches["draw_predicted"] == 0) & (valid_matches["is_draw"] == 0)
                        ).sum()
                        false_negatives = (
                            (valid_matches["draw_predicted"] == 0) & (valid_matches["is_draw"] == 1)
                        ).sum()

                        print("\nDetailed Metrics:")
                        print(f"Actual Draws: {valid_matches['is_draw'].sum()}")
                        print(f"Predicted Draws: {valid_matches['draw_predicted'].sum()}")

                        # Calculate metrics
                        accuracy = (true_positives + true_negatives) / len(matches_with_results)

                        if true_positives + false_negatives > 0:
                            draws_recall = true_positives / (true_positives + false_negatives)

                        if true_positives + false_positives > 0:
                            precision = true_positives / (true_positives + false_positives)

                        print("\nFinal Metrics:")
                        print(f"Accuracy: {accuracy:.2%}")
                        print(f"Precision: {precision:.2%}")
                        print(f"Recall: {draws_recall:.2%}")
                        print(f"Threshold: {predictor.threshold:.2%}")
                    else:
                        print("Warning: No match outcomes available for metric calculation")
            else:
                print("Warning: No real scores data available")
                matches_with_results = prediction_df.copy()
            matches_with_results = matches_with_results.loc[
                :, ~matches_with_results.columns.duplicated(keep="last")
            ]
        # Filter matches with results for date >= 2025-04-01 and order by date descending
        if "Date" in matches_with_results.columns:
            matches_with_results["Date"] = pd.to_datetime(matches_with_results["Date"])
            matches_with_results = matches_with_results[
                matches_with_results["Date"] >= "2025-04-01"
            ]
            matches_with_results = matches_with_results.sort_values(by="Date", ascending=False)
        return matches_with_results, precision, draws_recall
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return pd.DataFrame(), 0.0, 0.0


def apply_threshold_filter(df: pd.DataFrame, remove_thresholds: list[float]) -> pd.DataFrame:
    """
    Remove all rows from df where 'draw_probability' matches any of the remove_thresholds.
    Args:
        df (pd.DataFrame): DataFrame with predictions.
        remove_thresholds (List[float]): List of threshold values to remove.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # Create a mask for rows to keep (where draw_probability is not in remove_thresholds)
    mask = ~df["draw_probability"].isin(remove_thresholds)

    # Count number of rows being deleted for each threshold
    for threshold in remove_thresholds:
        num_deleted = (df["draw_probability"] == threshold).sum()
        print(f"Deleted {num_deleted} rows at threshold {threshold}")

    # Apply the filter and return the filtered DataFrame
    return df[mask]


def apply_keep_thresholds_filter(df: pd.DataFrame, allowed_thresholds: list[float]) -> pd.DataFrame:
    """
    Keep only rows in df where 'draw_probability' is one of the allowed_threshold values.
    Args:
        df (pd.DataFrame): DataFrame with predictions.
        allowed_thresholds (List[float]): List of allowed threshold values.
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    condition = df["draw_probability"].isin(allowed_thresholds)
    return df[condition]


def main():
    best_precision = 0
    best_model_uri = None
    best_predictions = pd.DataFrame()  # Initialize empty DataFrame
    predicted_df = pd.DataFrame()  # Initialize predicted_df
    # Model URIs to evaluate
    model_uris = [
        # "035abdf986654b1e8b551d0ce044c929",
        # "d3c066618b4d425fbb2ffff99a478238",
        # "97207cdaab54477fa267d8cd29ce35e9",
        # "835b997b8acd46f7a72ab5350451e427",
        # "538f96a0c783429f9f2e6967cc4693a2",
        # "1aec65aae580476b813fe97fee26e9e0",
        # "355e5d963cf644debba80148a3fcd430",
        # "ffacc3add86741a3a91f6115aae22b26",
        # "c3c7d0788ef14f8da4ad02d709e2bca9",
        # "d111479cd80548dd9dad5176f1dca58d",
        # "d437d40570524d18be10da4251fc3c2a",
        # "0ebd3bd3b1f3472d9871100270888c92",
        # "39f18a98d0004d48afd4da01f89be180",
        # "d408352aa7174d53985b62286ff3c3c2",
        # "1bbb837e359b43a79949b47a4c747893",
        # "6b0fe5666ce0464087b3dfc9fcab9f44",
        # "e870ebb733e04fd5853c4633cdda2409",
        # "468ce9f3a3ee4eda857506a0f12ce066",
        # "aa2badab5f4e444dac33e6bc4a917e40",
        # "aaf2b87d37924dd7900f1f67692c28ab",
        # "295e613f2ec644b4902aca3516dabd9d",
        # "e837c4a1541844c0a5ab903198426694",
        "72820da7182e45bd95fe9c8fd54a028d",
        "e9f0053b6c1d4e71bc4c9689bc7960d6",
    ]
    # Filter configuration to remove predictions near specific thresholds
    filter_config = {
        "8d80522037ae4a9790b72129c06851a4": {"remove_thresholds": [0.45, 0.47]},
        "d3c066618b4d425fbb2ffff99a478238": {
            "remove_thresholds": [0.59, 0.60, 0.64, 0.65, 0.66, 0.69]
        },
        "7c12f45bc2c442818cf09c497eef4176": {"remove_thresholds": [0.32, 0.33]},
        "58f6a2c94ced4c1a9c724d19224cca8c": {
            "remove_thresholds": [0.32, 0.35, 0.36, 0.40, 0.41, 0.42]
        },
        "835b997b8acd46f7a72ab5350451e427": {"remove_thresholds": [0.36, 0.31, 0.32]},
        "01d70371af714f87b0b15393a5ce6853": {"remove_thresholds": [0.37, 0.41]},
    }

    # Keep configuration to only allow predictions near specific thresholds
    keep_config = {
        "97207cdaab54477fa267d8cd29ce35e9": {"keep_thresholds": [0.31, 0.32, 0.34, 0.37]},
        "8d80522037ae4a9790b72129c06851a4": {"keep_thresholds": [0.47, 0.48]},
        "d3c066618b4d425fbb2ffff99a478238": {"keep_thresholds": [0.66, 0.68, 0.71, 0.72, 0.73]},
        "035abdf986654b1e8b551d0ce044c929": {"keep_thresholds": [0.61, 0.62, 0.65, 0.68, 0.72]},
        "538f96a0c783429f9f2e6967cc4693a2": {"keep_thresholds": [0.33, 0.40, 0.41, 0.43, 0.48]},
        "403c8c5eaaf442898594e45e6998cff4": {"keep_thresholds": [0.35, 0.37, 0.42]},
    }

    # Get preprocessed prediction data using standardized function
    prediction_df = create_prediction_set_ensemble()
    prediction_data = prediction_df.copy()
    print(f"Loaded {len(prediction_data)} matches for prediction")

    try:
        # Get real scores with error handling
        real_scores_df = get_real_api_scores_from_excel()
        print(f"real_scores_df: {len(real_scores_df)}")
    except Exception as e:
        print(f"Error processing fixture IDs: {str(e)}")
        real_scores_df = pd.DataFrame()

    # Evaluate each model
    for uri in model_uris:
        try:
            uri_full = f"runs:/{uri}/ensemble_model"
            predicted_df, precision, draws_recall = make_prediction(
                prediction_data, uri_full, real_scores_df
            )
            if not isinstance(predicted_df, pd.DataFrame) or predicted_df.empty:
                print(f"Skipping invalid predictions from model {uri}")
                continue

            # --- Apply remove threshold filtering if configured for this model ---
            config = filter_config.get(uri, None)
            if config is not None:
                remove_thresholds = config.get("remove_thresholds", None)
                if remove_thresholds is not None:
                    print(
                        f"Applying remove threshold filter for model {uri}: removing all predictions with draw_probability in {remove_thresholds}"
                    )
                    predicted_df = apply_threshold_filter(predicted_df, remove_thresholds)
            # --- Apply keep threshold filtering if configured for this model ---
            config_keep = keep_config.get(uri, None)
            if config_keep is not None:
                allowed_thresholds = config_keep.get("keep_thresholds", None)
                if allowed_thresholds is not None:
                    print(
                        f"Applying keep threshold filter for model {uri}: keeping only predictions with draw_probability in {allowed_thresholds}"
                    )
                    predicted_df = apply_keep_thresholds_filter(predicted_df, allowed_thresholds)
            # Remove rows where draw_predicted is 0
            predicted_df = predicted_df[predicted_df["draw_predicted"] == 1]
            print(f"Filtered to {len(predicted_df)} rows where draw_predicted = 1")
            # Save individual model predictions
            model_output_path = Path(f"./data/prediction/ensemble/predictions_model_{uri}.xlsx")
            # Reorder columns to place draw_predicted and draw_probability last
            cols = [
                col
                for col in predicted_df.columns
                if col not in ["draw_predicted", "draw_probability"]
            ]
            cols.extend(["draw_predicted", "draw_probability"])
            predicted_df = predicted_df[cols]
            predicted_df.to_excel(model_output_path, index=False)
            print(f"Predictions for model {uri} saved to: {model_output_path}")

            if precision > best_precision and draws_recall > 0.20:
                best_precision = precision
                best_model_uri = uri
                best_predictions = predicted_df.copy()
                print(f"New best model: {uri} with precision: {precision:.2%}")
                print(f"Draws recall: {draws_recall:.2%}")
        except Exception as e:
            print(f"Error evaluating model {uri}: {str(e)}")
            continue
    print(f"\nBest model URI: {best_model_uri}")
    print(f"Best precision: {best_precision:.2%}")

    # Handle empty predictions for best model
    if best_predictions.empty:
        print("Warning: No valid predictions generated. Creating empty result.")
        predicted_df = pd.DataFrame(columns=["fixture_id", "draw_predicted", "draw_probability"])
    else:
        predicted_df = best_predictions
        cols = [
            col for col in predicted_df.columns if col not in ["draw_predicted", "draw_probability"]
        ]
        cols.extend(["draw_predicted", "draw_probability"])
        predicted_df = predicted_df[cols]

    # Save best model results
    output_path = Path("./data/prediction/ensemble/predictions_ensemble_best.xlsx")
    predicted_df.to_excel(output_path, index=False)
    print(f"\nBest model predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
