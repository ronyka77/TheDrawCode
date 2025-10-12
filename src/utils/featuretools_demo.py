"""
Demonstration script for Featuretools Automated Feature Engineering.

This script shows how to use the SoccerFeaturetoolsEngineer to augment
existing features with automated temporal and relational feature generation.
"""

import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.StackedEnsemble.shared.data_loader_new import DataLoader
from src.utils.featuretools_automated_features import SoccerFeaturetoolsEngineer
from src.utils.logger import ExperimentLogger

logger = ExperimentLogger(experiment_name="featuretools_demo")



def load_real_data() -> pd.DataFrame:
    """
    Load real soccer match data using the project's DataLoader.
    This provides actual match data with all ~300 engineered features.
    """
    logger.info("Loading real soccer match data using DataLoader...")
    
    # Initialize the data loader
    data_loader = DataLoader(experiment_name="featuretools_real_data")
    
    # Load all data splits
    X_train, y_train, X_test, y_test, X_val, y_val = data_loader.load_data()
    
    # Combine all data for feature engineering demonstration
    # In production, you would typically only use training data for feature engineering
    logger.info("Combining all data splits for comprehensive feature engineering demo...")
    
    # Add target variable to each split for identification
    X_train_with_target = X_train.copy()
    X_train_with_target['target'] = y_train
    X_train_with_target['split'] = 'train'
    
    X_test_with_target = X_test.copy()
    X_test_with_target['target'] = y_test
    X_test_with_target['split'] = 'test'
    
    X_val_with_target = X_val.copy()
    X_val_with_target['target'] = y_val
    X_val_with_target['split'] = 'validation'
    
    # Combine all splits
    combined_df = pd.concat([
        X_train_with_target,
        X_test_with_target,
        X_val_with_target
    ], ignore_index=True)
    
    # Add required columns for featuretools if they don't exist
    if 'fixture_id' not in combined_df.columns:
        combined_df['fixture_id'] = range(len(combined_df))
        logger.info("Added fixture_id column (generated sequence)")
    
    if 'date_encoded' not in combined_df.columns:
        # Create a date sequence based on row order (assuming chronological order)
        combined_df['date_encoded'] = pd.date_range(
            start='2020-01-01', 
            periods=len(combined_df), 
            freq='D'
        )
        logger.info("Added date_encoded column (generated date sequence)")
    
    # Ensure proper data types for featuretools
    combined_df['fixture_id'] = combined_df['fixture_id'].astype(int)
    if 'home_encoded' in combined_df.columns:
        combined_df['home_encoded'] = combined_df['home_encoded'].astype(int)
    if 'away_encoded' in combined_df.columns:
        combined_df['away_encoded'] = combined_df['away_encoded'].astype(int)
    if 'venue_encoded' in combined_df.columns:
        combined_df['venue_encoded'] = combined_df['venue_encoded'].astype(int)
    if 'league_encoded' in combined_df.columns:
        combined_df['league_encoded'] = combined_df['league_encoded'].astype(int)
    
    logger.info(f"Loaded real dataset with {len(combined_df)} rows and {len(combined_df.columns)} features")
    logger.info(f"Data splits: Train={len(X_train)}, Test={len(X_test)}, Validation={len(X_val)}")
    logger.info(f"Draw rate: {combined_df['target'].mean():.2%}")
    
    # Log feature categories
    feature_names = data_loader.get_feature_names()
    logger.info(f"Available features from DataLoader: {len(feature_names)}")
    
    return combined_df


def load_sample_real_data(sample_size: int = 1000) -> pd.DataFrame:
    """
    Load a sample of real data for faster demonstration.
    
    Args:
        sample_size: Number of samples to use for demo
        
    Returns:
        Sampled dataframe with real features
    """
    logger.info(f"Loading sample of {sample_size} rows from real data for demo...")
    
    # Load full real data
    full_df = load_real_data()
    
    # Sample for faster processing
    if len(full_df) > sample_size:
        sampled_df = full_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled {sample_size} rows from {len(full_df)} total rows")
    else:
        sampled_df = full_df
        logger.info(f"Using all {len(full_df)} rows (less than requested sample size)")
    
    return sampled_df


def demonstrate_featuretools_pipeline():
    """Demonstrate the complete featuretools feature engineering pipeline."""
    
    logger.info("=== Featuretools Automated Feature Engineering Demo ===")
    
    # Step 1: Load real data (sampled for demo performance)
    logger.info("Step 1: Loading real soccer match data (sampled for demo)...")
    df = load_sample_real_data(sample_size=2000)  # Use 2000 samples for demo
    original_feature_count = len(df.columns)
    
    # Step 2: Initialize the feature engineer
    logger.info("Step 2: Initializing SoccerFeaturetoolsEngineer...")
    engineer = SoccerFeaturetoolsEngineer(
        logger=logger,
        mlflow_tracking=True,
        max_depth=2,
        n_jobs=1  # CPU-only constraint from project requirements
    )
    
    # Step 3: Run hybrid feature engineering
    logger.info("Step 3: Running hybrid feature engineering...")
    
    try:
        # Start MLflow experiment
        mlflow.set_experiment("featuretools_soccer_demo")
        
        with mlflow.start_run(run_name="hybrid_feature_engineering"):
            # Log original data info
            mlflow.log_params({
                'original_features': original_feature_count,
                'data_samples': len(df),
                'approach': 'hybrid_temporal_relational_interactions'
            })
            
            # Run feature engineering
            augmented_df, feature_defs = engineer.run_hybrid_feature_engineering(
                df=df,
                include_temporal=True,
                include_relational=True,
                include_interactions=True,
                temporal_window=5,
                temporal_gap=1
            )
            
            # Calculate results
            new_feature_count = len(augmented_df.columns)
            added_features = new_feature_count - original_feature_count
            
            # Log results
            mlflow.log_metrics({
                'original_features': original_feature_count,
                'new_features': new_feature_count,
                'added_features': added_features,
                'feature_increase_ratio': added_features / original_feature_count
            })
            
            logger.info(f"   Feature engineering completed successfully!")
            logger.info(f"   Original features: {original_feature_count} (real engineered features)")
            logger.info(f"   New features: {new_feature_count}")
            logger.info(f"   Added features: {added_features}")
            logger.info(f"   Increase ratio: {added_features / original_feature_count:.2%}")
            logger.info(f"   Data source: Real soccer match data from DataLoader")
            
            # Step 4: Analyze new features
            logger.info("Step 4: Analyzing new features...")
            
            new_feature_names = [col for col in augmented_df.columns if col not in df.columns]
            
            if new_feature_names:
                logger.info("New features generated:")
                
                # Group by feature type
                temporal_features = [f for f in new_feature_names if 'ft_temporal_' in f]
                relational_features = [f for f in new_feature_names if 'ft_relational_' in f]
                interaction_features = [f for f in new_feature_names if f.startswith('ft_') and 'temporal' not in f and 'relational' not in f]
                
                logger.info(f"   - Temporal features: {len(temporal_features)}")
                logger.info(f"   - Relational features: {len(relational_features)}")
                logger.info(f"   - Interaction features: {len(interaction_features)}")
                
                # Log feature examples
                if temporal_features:
                    logger.info(f"   Temporal examples: {temporal_features[:3]}")
                if relational_features:
                    logger.info(f"   Relational examples: {relational_features[:3]}")
                if interaction_features:
                    logger.info(f"   Interaction examples: {interaction_features[:3]}")
                
                # Save feature definitions
                feature_def_path = "featuretools_feature_definitions.json"
                engineer.save_feature_definitions(feature_defs, feature_def_path)
                
                # Log artifact
                mlflow.log_artifact(feature_def_path)
                
            else:
                logger.warning("No new features were generated")
            
            # Step 5: Data quality checks
            logger.info("Step 5: Performing data quality checks...")
            
            # Check for missing values in new features
            if new_feature_names:
                missing_stats = augmented_df[new_feature_names].isnull().sum()
                high_missing = missing_stats[missing_stats > len(augmented_df) * 0.1]
                
                if len(high_missing) > 0:
                    logger.warning(f"Features with >10% missing values: {len(high_missing)}")
                else:
                    logger.info("No features with excessive missing values")
                
                # Check for infinite values
                inf_count = np.isinf(augmented_df[new_feature_names].select_dtypes(include=[np.number])).sum().sum()
                if inf_count > 0:
                    logger.warning(f"Found {inf_count} infinite values in new features")
                else:
                    logger.info("No infinite values found in new features")
                
                # Basic statistics
                logger.info("Feature statistics summary:")
                stats = augmented_df[new_feature_names].describe()
                logger.info(f"   Mean std: {stats.loc['std'].mean():.4f}")
                logger.info(f"   Zero variance features: {(stats.loc['std'] == 0).sum()}")
            
            logger.info("=== Demo completed successfully! ===")
            
            return augmented_df, feature_defs
            
    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {str(e)}")
        raise


def demonstrate_feature_integration():
    """Demonstrate how to integrate new features with existing ensemble models using intelligent selection."""
    
    logger.info("=== Enhanced Feature Integration Demo ===")
    
    # Load only training data for feature engineering (best practice)
    logger.info("Loading training data only for feature engineering...")
    data_loader = DataLoader(experiment_name="featuretools_integration")
    X_train, y_train, _, _, _, _ = data_loader.load_data()
    
    # Add required columns for featuretools if they don't exist
    if 'fixture_id' not in X_train.columns:
        X_train['fixture_id'] = range(len(X_train))
    if 'date_encoded' not in X_train.columns:
        X_train['date_encoded'] = pd.date_range(start='2020-01-01', periods=len(X_train), freq='D')
    
    # Add team and venue IDs if missing (for demonstration)
    for col, max_val in [('home_encoded', 21), ('away_encoded', 21), ('venue_encoded', 11), ('league_encoded', 6)]:
        if col not in X_train.columns:
            X_train[col] = np.random.randint(1, max_val, len(X_train))
    
    # Ensure proper data types
    for col in ['fixture_id', 'home_encoded', 'away_encoded', 'venue_encoded', 'league_encoded']:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(int)
    
    logger.info(f"Training data prepared: {X_train.shape}")
    
    # Run feature engineering on training data
    engineer = SoccerFeaturetoolsEngineer(logger=logger, mlflow_tracking=False)
    
    augmented_df, feature_defs = engineer.run_hybrid_feature_engineering(
        df=X_train,
        include_temporal=True,
        include_relational=True,
        include_interactions=True
    )
    
    # Get new features
    new_features = [col for col in augmented_df.columns if col not in X_train.columns]
    
    if new_features:
        logger.info(f"Generated {len(new_features)} new features for ensemble integration")
        
        # Enhanced feature selection using multiple evaluation methods
        logger.info("Evaluating feature importance using multiple methods...")
        
        # Evaluate features using different methods
        top_features_mi = engineer.evaluate_feature_importance(
            X=augmented_df, 
            y=y_train, 
            new_feature_names=new_features,
            method='mutual_info',
            top_k=50
        )
        
        top_features_corr = engineer.evaluate_feature_importance(
            X=augmented_df, 
            y=y_train, 
            new_feature_names=new_features,
            method='correlation',
            top_k=50
        )
        
        top_features_combined = engineer.evaluate_feature_importance(
            X=augmented_df, 
            y=y_train, 
            new_feature_names=new_features,
            method='combined',
            top_k=100  # Get more for model-specific selection
        )
        
        logger.info(f"Top features by mutual info: {len(top_features_mi)}")
        logger.info(f"Top features by correlation: {len(top_features_corr)}")
        logger.info(f"Top features by combined score: {len(top_features_combined)}")
        
        # Load existing feature selections
        with open('src/utils/selected_features_ensemble_new.json', 'r') as f:
            existing_selections = json.load(f)
        
        # Create updated feature selections with intelligent selection
        updated_selections = existing_selections.copy()
        
        # Model-specific feature selection strategy
        model_feature_strategies = {
            'xgb': top_features_combined[:30],  # XGBoost handles many features well
            'catboost': top_features_mi[:25],   # CatBoost + mutual info for categorical handling
            'mlp': top_features_corr[:20],      # Neural networks + correlation for linear relationships
            'pytorch': top_features_combined[:35]  # PyTorch can handle complex interactions
        }
        
        # Add strategically selected features to each model
        for model_name, selected_features in model_feature_strategies.items():
            if model_name in updated_selections:
                # Remove duplicates while preserving order
                new_model_features = []
                existing_model_features = set(updated_selections[model_name])
                
                for feature in selected_features:
                    if feature not in existing_model_features:
                        new_model_features.append(feature)
                        existing_model_features.add(feature)
                
                updated_selections[model_name].extend(new_model_features)
                logger.info(f"Added {len(new_model_features)} strategically selected features to {model_name}")
                logger.info(f"   {model_name} examples: {new_model_features[:3]}")
        
        # Update 'all' selection with all top features (removing duplicates)
        all_top_features = list(dict.fromkeys(top_features_combined))  # Remove duplicates while preserving order
        updated_selections['all'].extend(all_top_features)
        
        # Save updated selections with metadata
        output_data = {
            'feature_selections': updated_selections,
            'metadata': {
                'generation_timestamp': pd.Timestamp.now().isoformat(),
                'total_generated_features': len(new_features),
                'features_selected_for_models': len(all_top_features),
                'selection_methods': ['mutual_info', 'correlation', 'combined'],
                'model_strategies': {
                    'xgb': 'Combined score (handles many features)',
                    'catboost': 'Mutual information (categorical handling)',
                    'mlp': 'Correlation (linear relationships)',
                    'pytorch': 'Combined score (complex interactions)'
                }
            }
        }
        
        output_path = "enhanced_selected_features_with_featuretools.json"
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Enhanced feature selections saved to {output_path}")
        logger.info(f"Total features in 'all': {len(updated_selections['all'])}")
        logger.info(f"Selection efficiency: {len(all_top_features)}/{len(new_features)} = {len(all_top_features)/len(new_features)*100:.1f}%")
        
        # Create feature analysis report
        feature_analysis = {
            'temporal_features': [f for f in new_features if 'ft_temporal_' in f],
            'relational_features': [f for f in new_features if 'ft_relational_' in f],
            'interaction_features': [f for f in new_features if f.startswith('ft_') and 'temporal' not in f and 'relational' not in f],
            'selected_temporal': [f for f in all_top_features if 'ft_temporal_' in f],
            'selected_relational': [f for f in all_top_features if 'ft_relational_' in f],
            'selected_interaction': [f for f in all_top_features if f.startswith('ft_') and 'temporal' not in f and 'relational' not in f]
        }
        
        logger.info("Feature type analysis:")
        for feature_type, features in feature_analysis.items():
            if 'selected_' in feature_type:
                original_type = feature_type.replace('selected_', '')
                if original_type in feature_analysis:
                    original_count = len(feature_analysis[original_type])
                    selected_count = len(features)
                    if original_count > 0:
                        logger.info(f"   {feature_type}: {selected_count}/{original_count} ({selected_count/original_count*100:.1f}%)")
                    else:
                        logger.info(f"   {feature_type}: {selected_count}/0 (no original features of this type)")
        
    else:
        logger.warning("No new features generated for integration")


if __name__ == "__main__":
    # Run the demonstrations
    try:
        # Main feature engineering demo
        augmented_df, feature_defs = demonstrate_featuretools_pipeline()
        
        # Feature integration demo
        demonstrate_feature_integration()
        
        print("\n" + "="*60)
        print("🎉 Featuretools demonstration completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("1. Integrate new features into your ensemble models")
        print("2. Evaluate model performance with augmented features")
        print("3. Use feature importance to select best new features")
        print("4. Update your training pipeline with the new features")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        sys.exit(1) 