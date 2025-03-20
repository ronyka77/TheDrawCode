import unittest
from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent

    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root create_evaluation_set: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent)
    print(f"Current directory create_evaluation_set: {os.getcwd().parent}")

from utils.create_evaluation_set import get_selected_api_columns_draws, convert_numeric_columns


class TestCreateEvaluationSet(unittest.TestCase):
    def test_get_selected_api_columns_draws(self):
        """Test that the column list is correct, has the expected length, and contains no duplicates."""
        columns = get_selected_api_columns_draws()
        # Check the return type
        self.assertIsInstance(columns, list, "Expected columns to be a list.")

        # For the new set of columns provided in our updated function
        expected_length = 67  # Updated to match actual number of columns
        self.assertEqual(len(columns), expected_length, f"Expected {expected_length} columns, got {len(columns)}.")

        # Ensure there are no duplicates by comparing in a set
        self.assertEqual(len(set(columns)), len(columns), "Duplicate columns found in the returned list.")

        # Verify some known columns exist
        expected_columns = [
            'venue_draw_rate', 
            'home_defensive_activity',
            'home_draw_rate',
            'away_expected_goals',
            'form_weighted_xg_diff',
            'defensive_stability',
            'referee_encoded'
        ]
        for col in expected_columns:
            self.assertIn(col, columns, f"Expected column '{col}' missing from the list.")


class TestConvertNumericColumns(unittest.TestCase):
    """Test cases for the convert_numeric_columns utility function."""

    def setUp(self):
        """Set up test cases with sample data."""
        self.test_data = pd.DataFrame({
            'normal': ['1.5', '2.5', '3.0'],
            'comma_decimal': ['1,5', '2,5', '3,0'],
            'scientific': ['1e-10', '2e5', '3E-3'],
            'mixed': ['1.5', '2,5', '3e-1'],
            'with_spaces': [' 1.5 ', ' 2.5', '3.0 '],
            'with_quotes': ["'1.5'", '"2.5"', '3.0'],
            'empty': ['', '', ''],
            'non_numeric': ['a', 'b', 'c'],
            'infinity': ['inf', '-inf', '1.5']
        })

    def test_basic_conversion(self):
        """Test basic numeric conversion functionality."""
        result = convert_numeric_columns(self.test_data, columns=['normal'])
        self.assertTrue(np.issubdtype(result['normal'].dtype, np.number))
        np.testing.assert_array_almost_equal(
            result['normal'].values,
            np.array([1.5, 2.5, 3.0])
        )

    def test_comma_decimal_conversion(self):
        """Test conversion of numbers with comma as decimal separator."""
        result = convert_numeric_columns(self.test_data, columns=['comma_decimal'])
        self.assertTrue(np.issubdtype(result['comma_decimal'].dtype, np.number))
        np.testing.assert_array_almost_equal(
            result['comma_decimal'].values,
            np.array([1.5, 2.5, 3.0])
        )

    def test_scientific_notation(self):
        """Test conversion of scientific notation numbers."""
        result = convert_numeric_columns(self.test_data, columns=['scientific'])
        self.assertTrue(np.issubdtype(result['scientific'].dtype, np.number))
        np.testing.assert_array_almost_equal(
            result['scientific'].values,
            np.array([1e-10, 2e5, 3e-3])
        )

    def test_mixed_formats(self):
        """Test conversion of mixed number formats."""
        result = convert_numeric_columns(self.test_data, columns=['mixed'])
        self.assertTrue(np.issubdtype(result['mixed'].dtype, np.number))
        np.testing.assert_array_almost_equal(
            result['mixed'].values,
            np.array([1.5, 2.5, 0.3])
        )

    def test_whitespace_handling(self):
        """Test handling of whitespace in numeric strings."""
        result = convert_numeric_columns(self.test_data, columns=['with_spaces'])
        self.assertTrue(np.issubdtype(result['with_spaces'].dtype, np.number))
        np.testing.assert_array_almost_equal(
            result['with_spaces'].values,
            np.array([1.5, 2.5, 3.0])
        )

    def test_quote_handling(self):
        """Test handling of quoted numeric strings."""
        result = convert_numeric_columns(self.test_data, columns=['with_quotes'])
        self.assertTrue(np.issubdtype(result['with_quotes'].dtype, np.number))
        np.testing.assert_array_almost_equal(
            result['with_quotes'].values,
            np.array([1.5, 2.5, 3.0])
        )

    def test_empty_string_handling(self):
        """Test handling of empty strings."""
        test_df = pd.DataFrame({
            'empty': ['', '', '']
        })
        result = convert_numeric_columns(
            test_df,
            columns=['empty'],
            drop_errors=True,
            verbose=True
        )
        # Empty strings should be treated as non-numeric and dropped
        self.assertNotIn('empty', result.columns)
        self.assertEqual(len(result.columns), 0)

    def test_non_numeric_handling(self):
        """Test handling of non-numeric strings."""
        # Create a DataFrame with only the non-numeric column
        test_df = pd.DataFrame({
            'non_numeric': ['a', 'b', 'c']
        })
        
        result = convert_numeric_columns(
            test_df,
            columns=['non_numeric'],
            drop_errors=True,
            verbose=True
        )
        
        # Check that the column was dropped
        self.assertNotIn('non_numeric', result.columns)
        self.assertEqual(len(result.columns), 0)

    def test_infinity_handling(self):
        """Test handling of infinity values."""
        result = convert_numeric_columns(self.test_data, columns=['infinity'])
        self.assertTrue(np.issubdtype(result['infinity'].dtype, np.number))
        np.testing.assert_array_almost_equal(
            result['infinity'].values,
            np.array([0.0, 0.0, 1.5])  # inf values should be replaced with 0.0
        )

    def test_all_columns_conversion(self):
        """Test conversion of all columns when no specific columns are specified."""
        result = convert_numeric_columns(
            self.test_data[['normal', 'comma_decimal']],
            columns=None
        )
        self.assertTrue(all(np.issubdtype(dtype, np.number)
                      for dtype in result.dtypes))
        self.assertEqual(len(result.columns), 2)


if __name__ == '__main__':
    unittest.main() 