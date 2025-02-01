import unittest
from pathlib import Path
import sys
import os

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

from utils.create_evaluation_set import get_selected_api_columns_draws


class TestCreateEvaluationSet(unittest.TestCase):
    def test_get_selected_api_columns_draws(self):
        """Test that the column list is correct, has the expected length, and contains no duplicates."""
        columns = get_selected_api_columns_draws()
        # Check the return type
        self.assertIsInstance(columns, list, "Expected columns to be a list.")

        # For the new set of columns provided in our updated function (e.g. 59 columns)
        expected_length = 59
        self.assertEqual(len(columns), expected_length, f"Expected {expected_length} columns, got {len(columns)}.")

        # Ensure there are no duplicates by comparing in a set
        self.assertEqual(len(set(columns)), len(columns), "Duplicate columns found in the returned list.")

        # Verify some known columns exist
        expected_columns = [
            'venue_draw_rate', 
            'form_weighted_xg_diff', 
            'referee_encoded',  # Newly introduced column in the replacement list.
            'draw_probability_score',  # Also part of updated columns.
            'Date',  # Expecting a column with the name 'Date'
            'Away_passes_mean'  # Ensure the corrected name.
        ]
        for col in expected_columns:
            self.assertIn(col, columns, f"Expected column '{col}' missing from the list.")

if __name__ == '__main__':
    unittest.main() 