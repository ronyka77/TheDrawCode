#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for model_training.py

This script tests the functions in model_training.py using the PyCaret library.

Usage:
    python -m unittest test_model_training.py

Author: AI Assistant
Date: 2023-10-15
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import functions to test
from models.pycaret.model_training import (
    setup_pycaret_environment,
    compare_models_for_precision,
    tune_model_for_precision,
    create_ensemble_model,
    evaluate_model_on_holdout
)
from models.pycaret.mlflow_module import save_model_and_predictions
from models.pycaret.feature_engineering import get_feature_importance
from models.pycaret.threshold_utils import precision_focused_score

class TestModelTraining(unittest.TestCase):
    """Test cases for model_training.py functions."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        # Create a simple synthetic dataset for testing
        np.random.seed(42)
        n_samples = 1000
        
        # Features
        X = np.random.randn(n_samples, 5)
        
        # Target (binary classification with class imbalance)
        y = np.zeros(n_samples)
        y[:int(n_samples * 0.2)] = 1  # 20% positive class
        
        # Create DataFrame
        cls.data = pd.DataFrame(
            data=X,
            columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        )
        cls.data['target'] = y
        
        # Add categorical features
        cls.data['cat_feature1'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
        cls.data['cat_feature2'] = np.random.choice(['X', 'Y', 'Z'], size=n_samples)
        
        # Create a temporary directory for saving outputs
        cls.temp_dir = "test_outputs"
        os.makedirs(cls.temp_dir, exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Remove temporary files
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
    
    def test_custom_precision_recall_scorer(self):
        """Test the custom scorer function."""
        # Create a simple test case
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1])
        
        # Test with different thresholds
        scorer = precision_focused_score(target_precision=0.6, min_recall=0.2)
        score = scorer(y_true, y_pred)
        
        # The score should be a float
        self.assertIsInstance(score, float)
    
    @patch('pycaret.classification.setup')
    def test_setup_pycaret_environment(self, mock_setup):
        """Test the setup_pycaret_environment function."""
        # Mock the setup function to return a MagicMock
        mock_setup.return_value = MagicMock()
        
        # Call the function
        result = setup_pycaret_environment(
            data=self.data,
            target_column='target',
            train_size=0.7,
            fold=5,
            normalize=True,
            feature_selection=True,
            fix_imbalance=True,
            session_id=42
        )
        
        # Check that setup was called
        mock_setup.assert_called_once()
        
        # Check that the result is the mock object
        self.assertEqual(result, mock_setup.return_value)
    
    @patch('pycaret.classification.setup')
    @patch('pycaret.classification.compare_models')
    def test_compare_models_for_precision(self, mock_compare_models, mock_setup):
        """Test the compare_models_for_precision function."""
        # Mock setup to return a MagicMock
        mock_setup.return_value = MagicMock()
        
        # Mock compare_models to return a list of models
        mock_models = [MagicMock(), MagicMock(), MagicMock()]
        mock_compare_models.return_value = mock_models
        
        # Call the function
        result = compare_models_for_precision(
            include=['lightgbm', 'xgboost'],
            n_select=3,
            sort='Precision'
        )
        
        # Check that compare_models was called
        mock_compare_models.assert_called_once()
        
        # Check that the result is the mock models
        self.assertEqual(result, mock_models)
    
    @patch('pycaret.classification.setup')
    @patch('pycaret.classification.tune_model')
    def test_tune_model_for_precision(self, mock_tune_model, mock_setup):
        """Test the tune_model_for_precision function."""
        # Mock setup to return a MagicMock
        mock_setup.return_value = MagicMock()
        
        # Mock tune_model to return a model
        mock_model = MagicMock()
        mock_tune_model.return_value = mock_model
        
        # Call the function
        result = tune_model_for_precision(
            model=MagicMock(),
            optimize='Precision',
            n_iter=10
        )
        
        # Check that tune_model was called
        mock_tune_model.assert_called_once()
        
        # Check that the result is the mock model
        self.assertEqual(result, mock_model)
    
    @patch('pycaret.classification.setup')
    @patch('pycaret.classification.ensemble_model')
    def test_create_ensemble_model(self, mock_ensemble_model, mock_setup):
        """Test the create_ensemble_model function."""
        # Mock setup to return a MagicMock
        mock_setup.return_value = MagicMock()
        
        # Mock ensemble_model to return a model
        mock_model = MagicMock()
        mock_ensemble_model.return_value = mock_model
        
        # Call the function
        result = create_ensemble_model(
            estimator_list=[MagicMock(), MagicMock()],
            method='Stacking',
            optimize='Precision'
        )
        
        # Check that ensemble_model was called
        mock_ensemble_model.assert_called_once()
        
        # Check that the result is the mock model
        self.assertEqual(result, mock_model)
    
    @patch('pycaret.classification.setup')
    @patch('pycaret.classification.predict_model')
    def test_evaluate_model_on_holdout(self, mock_predict_model, mock_setup):
        """Test the evaluate_model_on_holdout function."""
        # Mock setup to return a MagicMock
        mock_setup.return_value = MagicMock()
        
        # Mock predict_model to return a DataFrame with metrics
        mock_df = pd.DataFrame({
            'Precision': [0.8],
            'Recall': [0.7],
            'F1': [0.75],
            'AUC': [0.85]
        })
        mock_predict_model.return_value = mock_df
        
        # Call the function
        result = evaluate_model_on_holdout(MagicMock())
        
        # Check that predict_model was called
        mock_predict_model.assert_called_once()
        
        # Check that the result contains the expected metrics
        self.assertIn('Precision', result)
        self.assertIn('Recall', result)
        self.assertIn('F1', result)
        self.assertIn('AUC', result)
    
    @patch('pycaret.classification.setup')
    @patch('pycaret.classification.save_model')
    @patch('pycaret.classification.predict_model')
    def test_save_model_and_predictions(self, mock_predict_model, mock_save_model, mock_setup):
        """Test the save_model_and_predictions function."""
        # Mock setup to return a MagicMock
        mock_setup.return_value = MagicMock()
        
        # Mock predict_model to return a DataFrame
        mock_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'Score_1': [0.8, 0.6, 0.9]
        })
        mock_predict_model.return_value = mock_df
        
        # Mock save_model to return a path
        mock_path = os.path.join(self.temp_dir, "test_model")
        mock_save_model.return_value = mock_path
        
        # Call the function
        path, predictions = save_model_and_predictions(
            model=MagicMock(),
            output_dir=self.temp_dir
        )
        
        # Check that save_model was called
        mock_save_model.assert_called_once()
        
        # Check that predict_model was called
        mock_predict_model.assert_called_once()
        
        # Check that the path is correct
        self.assertEqual(path, mock_path)
        
        # Check that predictions is a DataFrame
        self.assertIsInstance(predictions, pd.DataFrame)
    
    @patch('pycaret.classification.setup')
    @patch('pycaret.classification.get_config')
    def test_get_feature_importance(self, mock_get_config, mock_setup):
        """Test the get_feature_importance function."""
        # Mock setup to return a MagicMock
        mock_setup.return_value = MagicMock()
        
        # Mock get_config to return a dictionary with X_train
        mock_get_config.return_value = {
            'X_train': pd.DataFrame({
                'feature1': [1, 2, 3],
                'feature2': [4, 5, 6]
            })
        }
        
        # Create a mock model with feature_importances_ attribute
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.7, 0.3])
        
        # Call the function
        result = get_feature_importance(mock_model)
        
        # Check that get_config was called
        mock_get_config.assert_called_once()
        
        # Check that the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check that the DataFrame has the expected columns
        self.assertIn('Feature', result.columns)
        self.assertIn('Importance', result.columns)

if __name__ == '__main__':
    unittest.main() 