# Environment Setup Guide

## Overview
This guide covers the setup of your development environment for the Soccer Prediction Project.

## Prerequisites

- Python 3.8+
- Conda or Miniconda
- Git
- Windows 10/11 (CPU-only training environment)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TheDrawCode.git
cd TheDrawCode
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yml
conda activate soccerpredictor_env
```

3. Set up MLflow:
```bash
mkdir -p mlruns
export MLFLOW_TRACKING_URI=file://$(pwd)/mlruns
```

## Detailed Setup

### 1. Environment Configuration

The project uses conda for environment management. The environment is defined in `environment.yml`:

```yaml
name: soccerpredictor_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pip
  - numpy
  - pandas
  - scikit-learn
  - xgboost
  - mlflow
  - pip:
    - pycaret
    - featuretools
    - icecream
```

### 2. Project Structure Setup

Run the provided script to create the project structure:
```bash
chmod +x create_project_structure.sh
./create_project_structure.sh
```

### 3. Code Quality Tools

Install and configure Prospector:
```bash
pip install prospector[with_everything]
prospector --profile .prospector.yaml
```

### 4. MLflow Configuration

1. Create configuration file:
```yaml
# config/experiment_config.yaml
mlflow:
  tracking_uri: file:///$(pwd)/mlruns
  experiment_name: soccer_prediction
  artifact_location: ./mlruns
```

2. Set environment variables:
```bash
export GIT_PYTHON_GIT_EXECUTABLE="C:/Program Files/Git/bin/git.exe"
```

## Common Issues

### 1. Package Conflicts

If you encounter package conflicts:
```bash
conda install -c conda-forge --strict-channel-priority <package>
```

### 2. MLflow Tracking

If MLflow tracking fails:
```bash
# Check tracking URI
python -c "import mlflow; print(mlflow.get_tracking_uri())"

# Clear and reinitialize
rm -rf mlruns/*
mkdir -p mlruns
```

### 3. CUDA Issues

For CPU-only training:
```python
# In model configuration
params = {
    'tree_method': 'hist',  # Use CPU-based histogram method
    'device': 'cpu'
}
```

## Development Tools

### 1. VS Code Extensions
- Python
- Pylance
- MLflow
- Git Graph

### 2. Jupyter Setup
```bash
conda install jupyter
python -m ipykernel install --user --name=soccerpredictor_env
```

### 3. Debugging Tools
```bash
pip install icecream  # Enhanced print debugging
pip install ipdb     # Interactive debugging
```

## Verification

Run these commands to verify your setup:

```bash
# Check Python environment
python --version
conda list

# Test MLflow
python -c "import mlflow; mlflow.start_run()"

# Run code quality checks
prospector

# Run tests
python -m pytest python_tests/
```

## Additional Resources

- [Conda Documentation](https://docs.conda.io/)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [PyCaret Documentation](https://pycaret.org/) 