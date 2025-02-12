import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import logging
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import PoissonRegressor

class PoissonXGCalculator:
    """Enhanced Poisson Expected Goals (xG) Calculator for soccer matches."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.model_dir = "./pipeline/models/"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Feature groups for model training and prediction
        self.form_features = [
            'home_goal_rollingaverage', 'away_goal_rollingaverage',
            'home_xG_rolling_rollingaverage', 'away_xG_rolling_rollingaverage',
            'home_form_momentum', 'away_form_momentum',
            'home_goal_difference_rollingaverage', 'away_goal_difference_rollingaverage',
            'home_shot_on_target_rollingaverage', 'away_shot_on_target_rollingaverage',
            'home_shots_on_target_accuracy_rollingaverage', 'away_shots_on_target_accuracy_rollingaverage'
        ]
        
        self.team_quality_features = [
            'home_attack_strength', 'away_attack_strength',
            'home_defense_weakness', 'away_defense_weakness',
            'home_league_position', 'away_league_position',
            'Home_possession_mean', 'away_possession_mean',
            'Home_shot_on_target_mean', 'away_shot_on_target_mean'
        ]
        
        self.historical_features = [
            'home_win_rate', 'away_win_rate',
            'home_average_points', 'away_average_points',
            'Home_goal_difference_cum', 'Away_goal_difference_cum',
            'Home_points_cum', 'Away_points_cum',
            'home_draw_rate', 'away_draw_rate'
        ]
        
        self.all_features = (self.form_features + self.team_quality_features + 
                            self.historical_features)
        
        self.scaler = StandardScaler()
        self.home_model = None
        self.away_model = None
        
    def _validate_data(self, df: pd.DataFrame, is_training: bool = True) -> None:
        """Validate input data for required columns and quality."""
        # Check required features
        missing_features = [col for col in self.all_features if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        # Additional checks for training data
        if is_training:
            if 'home_goals' not in df.columns or 'away_goals' not in df.columns:
                raise ValueError("Training data must contain 'home_goals' and 'away_goals'")
            
            if len(df) < 100:  # Minimum required for reliable model fitting
                raise ValueError("Insufficient training data (minimum 100 rows required)")
        
        # Check for excessive missing values
        missing_pct = df[self.all_features].isnull().mean()
        problematic_cols = missing_pct[missing_pct > 0.1].index
        if not problematic_cols.empty:
            self.logger.warning(f"Columns with >10% missing values: {problematic_cols}")
            
    def _prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Prepare features for model training or prediction."""
        try:
            # Create feature matrix and add grouping columns temporarily
            feature_cols = self.all_features.copy()
            grouping_cols = ['league_encoded', 'season_encoded']
            
            # Add grouping columns if they exist in the original dataframe
            for col in grouping_cols:
                if col in df.columns and col not in feature_cols:
                    feature_cols.append(col)
            
            X = df[feature_cols].copy()
            
            # Convert string numbers with commas to float
            for col in self.all_features:
                if X[col].dtype == 'object':
                    try:
                        # Replace commas with periods and convert to float
                        X[col] = X[col].str.replace(',', '.').astype(float)
                    except Exception as e:
                        self.logger.warning(f"Could not convert column {col} to numeric: {str(e)}")
            
            # Handle missing values
            for col in self.all_features:  # Only process actual features
                missing_count = X[col].isnull().sum()
                if missing_count > 0:
                    self.logger.info(f"Handling missing values in {col} (count: {missing_count})")
                    
                    try:
                        # Try league and season based imputation first
                        if all(gcol in X.columns for gcol in grouping_cols):
                            group_median = X.groupby(grouping_cols)[col].transform('median')
                            X[col] = X[col].fillna(group_median)
                        
                        # If still has missing values, try season-based only
                        if X[col].isnull().any() and 'season_encoded' in X.columns:
                            season_median = X.groupby('season_encoded')[col].transform('median')
                            X[col] = X[col].fillna(season_median)
                        
                        # If still has missing values, use global median
                        if X[col].isnull().any():
                            X[col] = X[col].fillna(X[col].median())
                        
                        # Final fallback to 0
                        X[col] = X[col].fillna(0)
                        
                        remaining_nulls = X[col].isnull().sum()
                        if remaining_nulls > 0:
                            self.logger.warning(f"Could not fill all missing values in {col}. Remaining nulls: {remaining_nulls}")
                            
                    except Exception as e:
                        self.logger.error(f"Error processing column {col}: {str(e)}")
                        # Fallback to simple median imputation
                        X[col] = X[col].fillna(X[col].median() if not X[col].isnull().all() else 0)
            
            # Remove grouping columns before scaling
            X = X[self.all_features]
            
            # Scale features
            if is_training:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
            
            X_scaled = pd.DataFrame(X_scaled, columns=self.all_features, index=X.index)
            
            # Add constant term for statsmodels
            X_scaled = sm.add_constant(X_scaled)
            
            return X_scaled
            
        except Exception as e:
            self.logger.error(f"Error in feature preparation: {str(e)}")
            raise
        
    def fit(self, df: pd.DataFrame) -> None:
        """Fit separate Poisson regression models for home and away goals."""
        try:
            self.logger.info("Validating training data...")
            self._validate_data(df, is_training=True)
            
            self.logger.info("Preparing features for model fitting...")
            X = self._prepare_features(df, is_training=True)
            
            # Train home goals model with L2 regularization
            self.logger.info("Fitting home goals Poisson model...")
            self.home_model = PoissonRegressor(alpha=1.0, max_iter=1000).fit(X, df['home_goals'])
            
            # Train away goals model with L2 regularization
            self.logger.info("Fitting away goals Poisson model...")
            self.away_model = PoissonRegressor(alpha=1.0, max_iter=1000).fit(X, df['away_goals'])
            
            
            # Evaluate model performance
            home_predictions = self.home_model.predict(X)
            away_predictions = self.away_model.predict(X)
            
            # Calculate metrics
            home_mse = mean_squared_error(df['home_goals'], home_predictions)
            away_mse = mean_squared_error(df['away_goals'], away_predictions)
            home_r2 = r2_score(df['home_goals'], home_predictions)
            away_r2 = r2_score(df['away_goals'], away_predictions)
            
            self.logger.info("\nModel Performance Metrics:")
            self.logger.info("Home Goals Model:")
            self.logger.info(f"MSE: {home_mse:.4f}")
            self.logger.info(f"RMSE: {np.sqrt(home_mse):.4f}")
            self.logger.info(f"R2 Score: {home_r2:.4f}")
            
            self.logger.info("\nAway Goals Model:")
            self.logger.info(f"MSE: {away_mse:.4f}")
            self.logger.info(f"RMSE: {np.sqrt(away_mse):.4f}")
            self.logger.info(f"R2 Score: {away_r2:.4f}")
            
            # Log model coefficients
            self.logger.info("\nHome Model Coefficients:")
            for feature, coef in zip(X.columns, self.home_model.coef_):
                self.logger.info(f"{feature}: {coef:.4f}")
            
            self.logger.info("\nAway Model Coefficients:")
            for feature, coef in zip(X.columns, self.away_model.coef_):
                self.logger.info(f"{feature}: {coef:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error in model fitting: {str(e)}")
            raise   
            
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate separate home and away xG predictions for matches."""
        try:
            self._validate_data(df, is_training=False)
            
            # Convert strings to numeric values
            for col in self.all_features:
                if df[col].dtype == 'object':
                    try:
                        # Replace commas with periods and convert to float
                        df[col] = df[col].str.replace(',', '.').astype(float)
                    except Exception as e:
                        self.logger.warning(f"Could not convert column {col} to numeric: {str(e)}")
            
            # Prepare features for prediction
            X = self._prepare_features(df, is_training=False)
            
            print(f"X shape: {X.shape}")
            
            # Generate predictions using separate models
            home_predictions = self.home_model.predict(X)
            away_predictions = self.away_model.predict(X)
            
            # Clip predictions to avoid overflow
            home_predictions = np.clip(home_predictions, -150, 150)
            away_predictions = np.clip(away_predictions, -150, 150)
            
            # Ensure predictions are non-negative
            df['home_poisson_xG'] = np.maximum(home_predictions, 0)
            df['away_poisson_xG'] = np.maximum(away_predictions, 0)
            
            # Round predictions to 3 decimal places for readability
            df['home_poisson_xG'] = df['home_poisson_xG'].round(3)
            df['away_poisson_xG'] = df['away_poisson_xG'].round(3)
            
            # Example: Check if feature values are within expected range
            for col in self.all_features:
                if df[col].max() > 105 or df[col].min() < -105:
                    self.logger.warning(f"Feature {col} has extreme values: min={df[col].min()}, max={df[col].max()}")
            
            # Log model parameters and feature statistics for debugging
            if hasattr(self.home_model, 'params'):
                self.logger.debug("Home Model Parameters:")
                self.logger.debug(self.home_model.params)
            else:
                self.logger.debug("Home Model does not have 'params' attribute.")
            
            if hasattr(self.away_model, 'params'):
                self.logger.debug("Away Model Parameters:")
                self.logger.debug(self.away_model.params)
            else:
                self.logger.debug("Away Model does not have 'params' attribute.")
            
            self.logger.debug("Feature Statistics:")
            self.logger.debug(df[self.all_features].describe())
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def add_poisson_xG(self, df: pd.DataFrame, type: str) -> pd.DataFrame:
         # Sort by date if available
        if 'Datum' in df.columns:
            df['Datum'] = pd.to_datetime(df['Datum'])
            df = df.sort_values('Datum')
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
        
        if type == 'training' or type == 'api_training':
            # Fit model on training data
            self.logger.info("Training model on historical data...")
            self.fit(df)
            
            # Save the trained model
            self.save_models()
        
        # Process all datasets
        datasets = {
            'training': ('./data_files/model_data_training_newPoisson.xlsx'),
            'training_new': ('./data_files/model_data_training_withPoisson.xlsx'),
            'prediction': ('./data_files/model_data_prediction_newPoisson.xlsx'),
            'merged': ('./data_files/merged_data_prediction_newPoisson.csv'),
            'api_prediction': ('./data_files/api_football_prediction_newPoisson.xlsx'),
            'api_training': ('./data_files/api_football_training_newPoisson.xlsx'),
            'api_future': ('./data_files/api_football_future_newPoisson.xlsx')
        }
        
        output_path = datasets[type]
        
        # Generate predictions
        self.logger.info(f"Generating predictions for {type} dataset...")
        df_with_xg = self.predict(df)
        
        # Export results
        try:
            if output_path.endswith('.xlsx'):
                df_with_xg.to_excel(output_path, index=False)
            else:
                df_with_xg.to_csv(output_path, index=False)
            self.logger.info(f"Exported {type} data to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to export {type} data: {str(e)}")
            # Try alternative format if Excel export fails
            if output_path.endswith('.xlsx'):
                alt_path = output_path.replace('.xlsx', '.csv')
                df_with_xg.to_csv(alt_path, index=False)
                self.logger.info(f"Exported {type} data to alternative format: {alt_path}")
    
        self.logger.info("Data processing completed successfully")
            
    def process_data(self):
        """Process all data files using a single trained model."""
        try:
            self.logger.info("Starting data processing...")
            
            # Load training data
            # training_path = './data_files/PowerBI/model_data_training.csv'
            # training_path_new = './data_files/PowerBI/model_data_training2.csv'
            # prediction_path = './data_files/PowerBI/model_data_prediction.csv'
            # merged_path = './data_files/PowerBI/merged_data_prediction.csv'
            api_prediction_path = './data_files/PowerBI/api_data_prediction.xlsx'
            api_training_path = './data_files/PowerBI/api_data_training.xlsx'
            api_future_path = './data_files/PowerBI/api_football_future.xlsx'
            
            # training_data = pd.read_csv(training_path)
            # training_data_new = pd.read_csv(training_path_new)
            # prediction_data = pd.read_csv(prediction_path)
            # merged_data = pd.read_csv(merged_path)
   

            # Read Excel files with openpyxl, treating 'Infinity' and similar strings as NaN
            self.logger.info(f"Loading prediction data from {api_prediction_path}")
            api_prediction_data = pd.read_excel(
                api_prediction_path,
                engine='openpyxl',
                na_values=['NaN', 'N/A', 'NA', 'null', 'None', '', 'Infinity', '-Infinity', 'inf', '-inf']
            ).replace([np.inf, -np.inf], np.nan)
            
            self.logger.info(f"Loading training data from {api_training_path}")
            api_training_data = pd.read_excel(
                api_training_path,
                engine='openpyxl',
                na_values=['NaN', 'N/A', 'NA', 'null', 'None', '', 'Infinity', '-Infinity', 'inf', '-inf']
            ).replace([np.inf, -np.inf], np.nan)
            
            self.logger.info(f"Loading future data from {api_future_path}")
            api_future_data = pd.read_excel(
                api_future_path,
                engine='openpyxl',
                na_values=['NaN', 'N/A', 'NA', 'null', 'None', '', 'Infinity', '-Infinity', 'inf', '-inf']
            ).replace([np.inf, -np.inf], np.nan)
            
            api_training_data = api_training_data.rename(columns={
                'home_possession_mean': 'Home_possession_mean',
                'home_shot_on_target_mean': 'Home_shot_on_target_mean',
                'away_goal_difference_cum': 'Away_goal_difference_cum',
                'home_points_cum': 'Home_points_cum',
                'away_points_cum': 'Away_points_cum'
            })
            api_prediction_data = api_prediction_data.rename(columns={
                'home_possession_mean': 'Home_possession_mean',
                'home_shot_on_target_mean': 'Home_shot_on_target_mean',
                'away_goal_difference_cum': 'Away_goal_difference_cum',
                'home_points_cum': 'Home_points_cum',
                'away_points_cum': 'Away_points_cum'
            })
            api_future_data = api_future_data.rename(columns={
                'home_possession_mean': 'Home_possession_mean',
                'home_shot_on_target_mean': 'Home_shot_on_target_mean',
                'away_goal_difference_cum': 'Away_goal_difference_cum',
                'home_points_cum': 'Home_points_cum',
                'away_points_cum': 'Away_points_cum'
            })
            
            # self.add_poisson_xG(training_data, 'training')
            # self.add_poisson_xG(training_data_new, 'training_new')
            # self.add_poisson_xG(prediction_data, 'prediction')
            # self.add_poisson_xG(merged_data, 'merged')
            self.add_poisson_xG(api_training_data, 'api_training')
            self.add_poisson_xG(api_prediction_data, 'api_prediction')
            self.add_poisson_xG(api_future_data, 'api_future')
           
            
        except Exception as e:
            self.logger.error(f"Error in data processing: {str(e)}")
            raise
            
    def save_models(self) -> None:
        """Save fitted models and scaler."""
        try:
            import joblib
            
            joblib.dump(self.home_model, os.path.join(self.model_dir, 'home_poisson_model.joblib'))
            joblib.dump(self.away_model, os.path.join(self.model_dir, 'away_poisson_model.joblib'))
            joblib.dump(self.scaler, os.path.join(self.model_dir, 'poisson_scaler.joblib'))
            
            self.logger.info("Models and scaler saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise
            
    def load_models(self) -> None:
        """Load saved models and scaler."""
        try:
            import joblib
            
            self.home_model = joblib.load(os.path.join(self.model_dir, 'home_poisson_model.joblib'))
            self.away_model = joblib.load(os.path.join(self.model_dir, 'away_poisson_model.joblib'))
            self.scaler = joblib.load(os.path.join(self.model_dir, 'poisson_scaler.joblib'))
            
            self.logger.info("Models and scaler loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise

def main():
    """Main execution function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./log/poisson_xg.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        calculator = PoissonXGCalculator(logger=logger)
        calculator.process_data()
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

def predict_new_data():
    """Prediction function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./log/poisson_xg_prediction.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        calculator = PoissonXGCalculator(logger=logger)
        calculator.load_models()
        prediction_data = pd.read_csv('./data_files/merged_data_prediction.csv')  # Replace with actual path
        prepared_data = calculator._prepare_features(prediction_data, is_training=False)
        
        home_goals_pred = calculator.home_model.predict(prepared_data)
        away_goals_pred = calculator.away_model.predict(prepared_data)
        
        prediction_data['home_poisson_xG'] = home_goals_pred
        prediction_data['away_poisson_xG'] = away_goals_pred

        prediction_data.to_csv('./data_files/merged_data_prediction_newPoisson.csv', index=False)  # Replace with actual path
        logger.info("Predictions saved successfully")
        
    except Exception as e:
        logger.error(f"Error in prediction execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()