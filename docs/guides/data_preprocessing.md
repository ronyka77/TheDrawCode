# Data Preprocessing Guide

## Numeric Data Conversion

### Overview

The `convert_numeric_columns` utility provides a robust solution for converting string-based numeric data into proper numeric types. It handles various formats, edge cases, and provides detailed error reporting.

### Features

- Comprehensive format support:
  - Standard decimal numbers
  - Scientific notation (e.g., '1e-10', '2E5')
  - Comma decimals (European format)
  - Mixed formats within columns
  - Quoted numbers
  - Numbers with whitespace

- Error handling:
  - Configurable error behavior (drop or keep problematic columns)
  - Detailed error reporting
  - NaN and infinity value handling
  - Empty string handling

- Performance optimization:
  - Efficient string operations
  - Vectorized operations where possible
  - Memory-efficient processing

### Usage

```python
from utils.create_evaluation_set import convert_numeric_columns

# Basic usage - convert all columns
df = convert_numeric_columns(data)

# Convert specific columns
df = convert_numeric_columns(
    data,
    columns=['col1', 'col2'],
    drop_errors=True,
    fill_value=0.0,
    verbose=True
)
```

### Parameters

- `data` (pd.DataFrame): Input DataFrame
- `columns` (Optional[List[str]]): Columns to convert (default: all columns)
- `drop_errors` (bool): Whether to drop columns that fail conversion
- `fill_value` (float): Value to use for empty strings/NaN
- `verbose` (bool): Whether to print conversion errors

### Examples

```python
# Handle mixed formats
df = pd.DataFrame({
    'mixed': ['1.5', '2,5', '3e-1'],
    'with_spaces': [' 1.5 ', ' 2.5', '3.0 ']
})
result = convert_numeric_columns(df)

# Handle problematic data
df = pd.DataFrame({
    'numbers': ['1.5', 'invalid', '3.0'],
    'empty': ['', '', '']
})
result = convert_numeric_columns(df, fill_value=-1)
```

### Error Handling

The function provides detailed error reporting:

```
Warning: Column col1 contains NaN values after conversion
Error converting column col2: could not convert string to float
Dropped column col2 due to conversion error

Conversion summary:
Failed columns: ['col2']
Successfully converted: 5 columns
Failed conversions: 1 columns
```

### Best Practices

1. **Data Validation**
   - Always check the conversion summary
   - Verify data types after conversion
   - Monitor NaN values in the result

2. **Error Handling**
   - Use verbose mode during development
   - Consider keeping problematic columns during debugging
   - Log conversion errors for analysis

3. **Performance**
   - Convert only necessary columns
   - Use appropriate fill values for your use case
   - Monitor memory usage with large datasets

### Integration

The utility is integrated into several data processing functions:

- `create_prediction_set_api()`
- `import_training_data_draws_api()`
- `create_evaluation_sets_draws_api()`

### Testing

A comprehensive test suite is available in `python_tests/test_create_evaluation_set.py`:

```bash
python -m unittest python_tests/test_create_evaluation_set.py
```

The test suite covers:
- Basic numeric conversion
- Comma decimal handling
- Scientific notation
- Mixed formats
- Edge cases
- Error conditions

### Related Documentation

- [Data Pipeline Guide](data_pipeline.md)
- [Feature Selection Guide](feature_selection.md)