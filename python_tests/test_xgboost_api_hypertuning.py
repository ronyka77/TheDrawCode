import unittest
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import sys
import os
import optuna
from unittest.mock import MagicMock, patch

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent
    if not project_root.exists():
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
except Exception as e:
    sys.path.append(os.getcwd())

from models.hypertuning.xgboost_api_hypertuning import GlobalHypertuner
from utils.logger import ExperimentLogger

class TestGlobalHypertuner(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.logger = MagicMock(spec=ExperimentLogger)
        self.hypertuner = GlobalHypertuner(logger=self.logger)
        
        # Create synthetic dataset
        np.random.seed(42)
        n_samples = 1500
        n_features = 59  # Matching the expected column count
        
        # Generate synthetic features
        self.X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.X_val = pd.DataFrame(
            np.random.randn(n_samples//2, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.X_test = pd.DataFrame(
            np.random.randn(n_samples//2, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Generate synthetic targets with realistic draw rates (~25%)
        self.y_train = pd.Series(np.random.binomial(1, 0.25, n_samples))
        self.y_val = pd.Series(np.random.binomial(1, 0.25, n_samples//2))
        self.y_test = pd.Series(np.random.binomial(1, 0.25, n_samples//2))

    def test_initialization(self):
        """Test that the hypertuner initializes with correct default values."""
        self.assertEqual(self.hypertuner.MIN_SAMPLES, 1000)
        self.assertEqual(self.hypertuner.DEFAULT_THRESHOLD, 0.53)
        self.assertEqual(self.hypertuner.TARGET_PRECISION, 0.5)
        self.assertEqual(self.hypertuner.TARGET_RECALL, 0.6)
        self.assertEqual(self.hypertuner.PRECISION_WEIGHT, 0.7)
        self.assertEqual(self.hypertuner.RECALL_CAP, 0.30)

    @patch('xgboost.XGBClassifier')
    def test_find_optimal_threshold(self, mock_xgb):
        """Test the optimal threshold finding logic."""
        # Mock XGBoost model predictions
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([
            [0.7, 0.3], [0.4, 0.6], [0.2, 0.8], [0.9, 0.1]
        ])
        
        # Test data
        X_val = pd.DataFrame({'feature': [1, 2, 3, 4]})
        y_val = pd.Series([1, 1, 0, 0])
        
        threshold, metrics = self.hypertuner._find_optimal_threshold(
            mock_model, X_val, y_val
        )
        
        self.assertIsInstance(threshold, float)
        self.assertTrue(0 <= threshold <= 1)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)

    def test_insufficient_samples(self):
        """Test that error is raised when sample size is too small."""
        small_X = pd.DataFrame(np.random.randn(500, 5))
        small_y = pd.Series(np.random.binomial(1, 0.25, 500))
        
        with self.assertRaises(ValueError):
            self.hypertuner.tune_global_model(
                small_X, small_y,
                small_X, small_y,
                small_X, small_y
            )

    @patch('mlflow.start_run')
    @patch('optuna.create_study')
    def test_tune_global_model(self, mock_study, mock_mlflow):
        """Test the complete tuning process."""
        # Mock study
        mock_trial = MagicMock()
        mock_trial.number = 1
        mock_trial.suggest_float.return_value = 0.1
        mock_trial.suggest_int.return_value = 100
        
        # Set up the study mock
        study = MagicMock()
        study.best_trial.params = {
            'learning_rate': 0.1,
            'min_child_weight': 100,
            'gamma': 5.0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1.0,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'n_estimators': 5000
        }
        mock_study.return_value = study
        
        try:
            best_params = self.hypertuner.tune_global_model(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                self.X_test, self.y_test,
                n_trials=1
            )
            
            self.assertIsInstance(best_params, dict)
            self.assertIn('learning_rate', best_params)
            self.assertIn('min_child_weight', best_params)
            
        except optuna.exceptions.TrialPruned:
            pass  # Trial pruning is expected behavior

    def test_evaluate_tuned_model(self):
        """Test model evaluation with tuned parameters."""
        # Set some mock best parameters
        self.hypertuner.best_params = {
            'learning_rate': 0.1,
            'min_child_weight': 100,
            'gamma': 5.0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1.0,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'n_estimators': 5000
        }
        
        metrics = self.hypertuner.evaluate_tuned_model(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.X_test, self.y_test
        )
        
        # Verify metrics structure
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('draw_rate', metrics)
        self.assertIn('predicted_rate', metrics)
        self.assertIn('n_samples', metrics)
        self.assertIn('n_draws', metrics)
        self.assertIn('n_predicted', metrics)
        self.assertIn('n_correct', metrics)
        self.assertIn('best_params', metrics)

    def test_evaluate_without_tuning(self):
        """Test that evaluation fails properly without tuned parameters."""
        self.hypertuner.best_params = {}
        
        with self.assertRaises(ValueError):
            self.hypertuner.evaluate_tuned_model(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                self.X_test, self.y_test
            )

if __name__ == '__main__':
    unittest.main() 