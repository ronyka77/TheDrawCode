"""
Automated Feature Engineering using Featuretools for Soccer Prediction.

This module implements a hybrid approach combining temporal and relational feature generation
to augment existing manually engineered features for improved soccer match prediction accuracy.
Optimized for memory efficiency and performance with central logger integration.
"""

import warnings
from typing import Optional, Union

import featuretools as ft
import numpy as np
import pandas as pd
from featuretools.primitives import (
    Count,
    Lag,
    Max,
    Mean,
    Min,
    RollingMax,
    RollingMean,
    RollingMin,
)

# Import central logger
from src.utils.logger import ExperimentLogger

# Suppress specific warnings that are expected/harmless
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
warnings.filterwarnings("ignore", module="woodwork", category=UserWarning)


class SoccerFeaturetoolsEngineer:
    """
    Automated feature engineering for soccer prediction using Featuretools.

    This class implements a hybrid approach that combines:
    1. Temporal feature synthesis for time-aware patterns
    2. Relational feature generation across entities (teams, matches, venues)
    3. Feature interaction discovery to complement existing features
    """

    def __init__(
        self,
        experiment_logger: ExperimentLogger,
        max_depth: int = 2,
        n_jobs: int = 1,  # CPU-only constraint
        chunk_size: Optional[int] = None,
        memory_limit_gb: float = 4.0,
    ):
        """
        Initialize the automated feature engineer with optimizations.

        Args:
            experiment_logger: Central ExperimentLogger instance
            max_depth: Maximum depth for deep feature synthesis
            n_jobs: Number of parallel jobs (kept at 1 for CPU-only constraint)
            chunk_size: Chunk size for large datasets (auto-calculated if None)
            memory_limit_gb: Memory limit for processing
        """
        self.logger = experiment_logger
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.chunk_size = chunk_size or self._calculate_optimal_chunk_size()
        self.memory_limit_gb = memory_limit_gb
        self.entityset = None
        self.feature_matrix = None
        self.feature_defs = None

        # Performance monitoring
        self._feature_generation_stats = {
            "temporal_features": 0,
            "relational_features": 0,
            "interaction_features": 0,
            "memory_usage_mb": 0,
        }

        # Define primitives for different feature types
        self._setup_primitives()

        self.logger.info(
            "SoccerFeaturetoolsEngineer initialized with optimizations",
            extra={
                "max_depth": max_depth,
                "chunk_size": self.chunk_size,
                "memory_limit_gb": memory_limit_gb,
            },
        )

    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on available memory."""
        try:
            # Conservative estimate: 1000 rows per GB of memory limit
            optimal_size = int(self.memory_limit_gb * 1000)
            return max(1000, min(optimal_size, 10000))  # Between 1K and 10K rows
        except:
            return 5000  # Safe default

    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency."""
        try:
            df_optimized = df.copy()

            # Optimize integer columns
            for col in df_optimized.select_dtypes(include=["int64"]).columns:
                col_min, col_max = df_optimized[col].min(), df_optimized[col].max()
                if col_min >= 0:
                    if col_max < 255:
                        df_optimized[col] = df_optimized[col].astype("uint8")
                    elif col_max < 65535:
                        df_optimized[col] = df_optimized[col].astype("uint16")
                    elif col_max < 4294967295:
                        df_optimized[col] = df_optimized[col].astype("uint32")
                else:
                    if col_min > -128 and col_max < 127:
                        df_optimized[col] = df_optimized[col].astype("int8")
                    elif col_min > -32768 and col_max < 32767:
                        df_optimized[col] = df_optimized[col].astype("int16")
                    elif col_min > -2147483648 and col_max < 2147483647:
                        df_optimized[col] = df_optimized[col].astype("int32")

            # Optimize float columns
            for col in df_optimized.select_dtypes(include=["float64"]).columns:
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast="float")

            return df_optimized
        except Exception as e:
            self.logger.warning(f"Data type optimization failed: {str(e)}")
            return df

    def _setup_primitives(self) -> None:
        """Setup primitive collections for different feature engineering approaches."""

        # Soccer-specific feature combinations
        self.soccer_feature_groups = {
            "attack_features": [
                "home_attack_strength",
                "away_attack_strength",
                "home_goal_rollingaverage",
                "away_goal_rollingaverage",
                "home_xG_rolling_rollingaverage",
                "away_xG_rolling_rollingaverage",
                "home_shot_on_target_rollingaverage",
                "away_shot_on_target_rollingaverage",
            ],
            "defense_features": [
                "home_defense_weakness",
                "away_defense_weakness",
                "home_saves_rollingaverage",
                "away_saves_rollingaverage",
                "defensive_stability",
            ],
            "form_features": [
                "home_form_momentum",
                "away_form_momentum",
                "form_stability",
                "form_difference",
                "home_xg_form",
                "away_xg_form",
            ],
            "tactical_features": [
                "Home_possession_mean",
                "away_possession_mean",
                "home_corners_rollingaverage",
                "away_corners_rollingaverage",
                "home_fouls_rollingaverage",
                "away_fouls_rollingaverage",
            ],
            "elo_features": ["home_team_elo", "away_team_elo", "elo_difference", "elo_similarity"],
        }

    def prepare_data_for_featuretools(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Prepare and structure data for featuretools entity creation with optimizations.

        Args:
            df: Input dataframe with soccer match data

        Returns:
            Dictionary of dataframes for different entities
        """
        try:
            initial_memory = df.memory_usage(deep=True).sum() / 1024**2
            self.logger.info(
                "Preparing data for featuretools entity creation",
                extra={"input_shape": df.shape, "input_memory_mb": initial_memory},
            )

            # OPTIMIZATION: Data type optimization first
            df_optimized = self._optimize_data_types(df.copy())

            # OPTIMIZATION: Categorical encoding for low-cardinality columns
            categorical_candidates = [
                "home_encoded",
                "away_encoded",
                "venue_encoded",
                "league_encoded",
            ]
            for col in categorical_candidates:
                if col in df_optimized.columns:
                    unique_ratio = df_optimized[col].nunique() / len(df_optimized)
                    if unique_ratio < 0.5:  # Less than 50% unique values
                        df_optimized[col] = df_optimized[col].astype("category")

            # Continue with existing logic using optimized dataframe
            matches_df = df_optimized

            # Ensure required columns exist
            required_cols = ["fixture_id", "date_encoded", "home_encoded", "away_encoded"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Create main matches entity
            matches_df = df.copy()

            # Create unique fixture_id to handle potential duplicates from combined data splits
            original_fixture_count = len(matches_df["fixture_id"].unique())
            total_rows = len(matches_df)

            if original_fixture_count != total_rows:
                self.logger.warning(
                    f"Non-unique fixture_ids detected: {original_fixture_count} unique vs {total_rows} total"
                )
                self.logger.info(
                    "Creating unique fixture_id sequence for featuretools compatibility"
                )

                # Store original fixture_id for reference
                matches_df["original_fixture_id"] = matches_df["fixture_id"].copy()

                # Create unique sequential fixture_id
                matches_df["fixture_id"] = range(len(matches_df))

                self.logger.info(
                    f"Generated unique fixture_id sequence: 0 to {len(matches_df) - 1}"
                )
            else:
                self.logger.info(
                    f"Fixture_id column is already unique ({original_fixture_count} matches)"
                )

            # Ensure proper data types
            matches_df["fixture_id"] = matches_df["fixture_id"].astype(int)
            matches_df["home_encoded"] = matches_df["home_encoded"].astype(int)
            matches_df["away_encoded"] = matches_df["away_encoded"].astype(int)

            # Convert date if needed with proper format handling
            if "date_encoded" in matches_df.columns:
                try:
                    # Check if date_encoded contains integer days since reference date
                    if matches_df["date_encoded"].dtype in ["int64", "int32", "float64"]:
                        # Convert integer days to actual dates (assuming reference date 2020-08-11)
                        reference_date = pd.to_datetime("2020-08-11", format="%Y-%m-%d")
                        matches_df["match_date"] = reference_date + pd.to_timedelta(
                            matches_df["date_encoded"], unit="D"
                        )
                        self.logger.info(
                            "Converted date_encoded (integer days) to match_date successfully"
                        )
                    elif matches_df["date_encoded"].dtype == "object":
                        # Try common date formats explicitly to avoid Woodwork warnings
                        date_formats = [
                            "%Y-%m-%d",
                            "%Y/%m/%d",
                            "%d/%m/%Y",
                            "%m/%d/%Y",
                            "%Y-%m-%d %H:%M:%S",
                        ]
                        matches_df["match_date"] = None

                        for fmt in date_formats:
                            try:
                                # Suppress warnings during format attempts
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    parsed_dates = pd.to_datetime(
                                        matches_df["date_encoded"], format=fmt, errors="coerce"
                                    )
                                if not parsed_dates.isna().all():
                                    matches_df["match_date"] = parsed_dates
                                    self.logger.info(
                                        f"Successfully parsed dates using format: {fmt}"
                                    )
                                    break
                            except:
                                continue

                        # If no format worked, try general parsing with warnings suppressed
                        if matches_df["match_date"].isna().all():
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                matches_df["match_date"] = pd.to_datetime(
                                    matches_df["date_encoded"], errors="coerce"
                                )
                                self.logger.info("Used general date parsing as fallback")
                    else:
                        # If it's already datetime, use as is
                        matches_df["match_date"] = pd.to_datetime(matches_df["date_encoded"])

                except Exception as e:
                    self.logger.warning(
                        f"Date conversion failed: {str(e)}, creating sequential dates"
                    )
                    # If all else fails, create a simple date sequence
                    matches_df["match_date"] = pd.date_range(
                        start="2020-01-01", periods=len(matches_df), freq="D"
                    )

                # Ensure no NaT values remain
                if matches_df["match_date"].isna().any():
                    self.logger.warning(
                        "Some dates could not be parsed, filling with sequential dates"
                    )
                    # Fill NaT values with sequential dates
                    base_date = pd.to_datetime("2020-01-01", format="%Y-%m-%d")
                    mask = matches_df["match_date"].isna()
                    # Use vectorized operations instead of iterating
                    sequential_dates = pd.date_range(start=base_date, periods=mask.sum(), freq="D")
                    matches_df.loc[mask, "match_date"] = sequential_dates

            # Create teams entity with validation
            home_teams = matches_df[["home_encoded"]].rename(columns={"home_encoded": "team_id"})
            away_teams = matches_df[["away_encoded"]].rename(columns={"away_encoded": "team_id"})
            teams_df = pd.concat([home_teams, away_teams]).drop_duplicates().reset_index(drop=True)

            # Validate teams entity
            if len(teams_df["team_id"].unique()) != len(teams_df):
                self.logger.warning(
                    "Duplicate team_ids detected in teams entity, removing duplicates"
                )
                teams_df = teams_df.drop_duplicates(subset=["team_id"]).reset_index(drop=True)

            self.logger.info(f"Created teams entity with {len(teams_df)} unique teams")

            # Create venues entity if venue data exists
            venues_df = None
            if "venue_encoded" in matches_df.columns:
                venue_cols = ["venue_encoded"]
                if "venue_capacity" in matches_df.columns:
                    venue_cols.append("venue_capacity")
                if "venue_draw_rate" in matches_df.columns:
                    venue_cols.append("venue_draw_rate")

                venues_df = matches_df[venue_cols].drop_duplicates()
                venues_df = venues_df.rename(columns={"venue_encoded": "venue_id"})
                venues_df = venues_df.dropna().reset_index(drop=True)

                # Validate venue entity uniqueness
                if len(venues_df) > 0 and len(venues_df["venue_id"].unique()) != len(venues_df):
                    self.logger.warning("Duplicate venue_ids detected, removing duplicates")
                    venues_df = venues_df.drop_duplicates(subset=["venue_id"]).reset_index(
                        drop=True
                    )

                self.logger.info(f"Created venues entity with {len(venues_df)} unique venues")

            # Create leagues entity if league data exists
            leagues_df = None
            if "league_encoded" in matches_df.columns:
                league_cols = ["league_encoded"]
                if "league_competitiveness" in matches_df.columns:
                    league_cols.append("league_competitiveness")
                if "league_draw_rate" in matches_df.columns:
                    league_cols.append("league_draw_rate")

                leagues_df = matches_df[league_cols].drop_duplicates()
                leagues_df = leagues_df.rename(columns={"league_encoded": "league_id"})
                leagues_df = leagues_df.dropna().reset_index(drop=True)

                # Validate league entity uniqueness
                if len(leagues_df) > 0 and len(leagues_df["league_id"].unique()) != len(leagues_df):
                    self.logger.warning("Duplicate league_ids detected, removing duplicates")
                    leagues_df = leagues_df.drop_duplicates(subset=["league_id"]).reset_index(
                        drop=True
                    )

                self.logger.info(f"Created leagues entity with {len(leagues_df)} unique leagues")

            entities = {
                "matches": matches_df,
                "teams": teams_df,
            }

            if venues_df is not None and len(venues_df) > 0:
                entities["venues"] = venues_df

            if leagues_df is not None and len(leagues_df) > 0:
                entities["leagues"] = leagues_df

            # OPTIMIZATION: Report memory savings
            final_memory = (
                sum(entity_df.memory_usage(deep=True).sum() for entity_df in entities.values())
                / 1024**2
            )
            memory_saved = initial_memory - final_memory

            self.logger.info(
                f"Created {len(entities)} entities for featuretools",
                extra={
                    "final_memory_mb": final_memory,
                    "memory_saved_mb": memory_saved,
                    "savings_percent": (memory_saved / initial_memory) * 100
                    if initial_memory > 0
                    else 0,
                },
            )
            return entities

        except Exception as e:
            self.logger.error(
                f"Error preparing data for featuretools: {str(e)}",
                extra={"function": "prepare_data_for_featuretools"},
            )
            raise

    def create_entityset(self, entities: dict[str, pd.DataFrame]) -> ft.EntitySet:
        """
        Create featuretools EntitySet with proper relationships.

        Args:
            entities: Dictionary of prepared dataframes

        Returns:
            Configured EntitySet
        """
        try:
            self.logger.info("Creating featuretools EntitySet")

            # Suppress Woodwork warnings during EntitySet creation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", message="Could not infer format")

                es = ft.EntitySet(id="soccer_matches")

                # Ensure match_date is properly formatted for Woodwork
                matches_df = entities["matches"].copy()
                if "match_date" in matches_df.columns:
                    # Ensure datetime format is standardized
                    matches_df["match_date"] = pd.to_datetime(
                        matches_df["match_date"], errors="coerce"
                    )
                    # Remove any remaining NaT values
                    if matches_df["match_date"].isna().any():
                        self.logger.warning(
                            "Removing rows with invalid dates for EntitySet creation"
                        )
                        matches_df = matches_df.dropna(subset=["match_date"])

                # Add matches entity (primary entity)
                es = es.add_dataframe(
                    dataframe=matches_df,
                    dataframe_name="matches",
                    index="fixture_id",
                    time_index="match_date" if "match_date" in matches_df.columns else None,
                    make_index=False,
                )

                # Add teams entity
                es = es.add_dataframe(
                    dataframe=entities["teams"],
                    dataframe_name="teams",
                    index="team_id",
                    make_index=False,
                )

                # Add relationships
                # Home team relationship
                es = es.add_relationship("teams", "team_id", "matches", "home_encoded")
                # Away team relationship
                es = es.add_relationship("teams", "team_id", "matches", "away_encoded")

                # Add venues entity and relationship if available
                if "venues" in entities and len(entities["venues"]) > 0:
                    es = es.add_dataframe(
                        dataframe=entities["venues"],
                        dataframe_name="venues",
                        index="venue_id",
                        make_index=False,
                    )
                    if "venue_encoded" in matches_df.columns:
                        es = es.add_relationship("venues", "venue_id", "matches", "venue_encoded")

                # Add leagues entity and relationship if available
                if "leagues" in entities and len(entities["leagues"]) > 0:
                    es = es.add_dataframe(
                        dataframe=entities["leagues"],
                        dataframe_name="leagues",
                        index="league_id",
                        make_index=False,
                    )
                    if "league_encoded" in matches_df.columns:
                        es = es.add_relationship(
                            "leagues", "league_id", "matches", "league_encoded"
                        )

            self.entityset = es
            self.logger.info(f"EntitySet created with {len(es.dataframes)} entities")
            return es

        except Exception as e:
            self.logger.error(f"Error creating EntitySet: {str(e)}")
            raise

    def generate_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate feature interactions specific to soccer prediction.

        Args:
            df: Input dataframe with existing features

        Returns:
            Dataframe with new interaction features
        """
        try:
            self.logger.info("Generating soccer-specific interaction features")

            result_df = df.copy()

            # Attack vs Defense interactions
            if all(col in df.columns for col in ["home_attack_strength", "away_defense_weakness"]):
                result_df["ft_home_attack_vs_away_defense"] = df["home_attack_strength"] * (
                    1 - df["away_defense_weakness"]
                )

            if all(col in df.columns for col in ["away_attack_strength", "home_defense_weakness"]):
                result_df["ft_away_attack_vs_home_defense"] = df["away_attack_strength"] * (
                    1 - df["home_defense_weakness"]
                )

            # Form momentum interactions
            if all(col in df.columns for col in ["home_form_momentum", "away_form_momentum"]):
                result_df["ft_form_momentum_difference"] = (
                    df["home_form_momentum"] - df["away_form_momentum"]
                )
                result_df["ft_form_momentum_product"] = (
                    df["home_form_momentum"] * df["away_form_momentum"]
                )

            # ELO-based interactions
            if all(col in df.columns for col in ["elo_difference", "form_difference"]):
                result_df["ft_elo_form_interaction"] = df["elo_difference"] * df["form_difference"]

            # Possession-based interactions
            if all(col in df.columns for col in ["Home_possession_mean", "away_possession_mean"]):
                result_df["ft_possession_dominance"] = (
                    df["Home_possession_mean"] - df["away_possession_mean"]
                )

            # xG momentum interactions
            if all(col in df.columns for col in ["home_xg_momentum", "away_xg_momentum"]):
                result_df["ft_xg_momentum_ratio"] = df["home_xg_momentum"] / (
                    df["away_xg_momentum"] + 1e-10
                )

            # League context interactions
            if all(col in df.columns for col in ["league_competitiveness", "elo_difference"]):
                result_df["ft_competitive_elo_interaction"] = df["league_competitiveness"] * abs(
                    df["elo_difference"]
                )

            # Rest advantage interactions
            if all(col in df.columns for col in ["home_rest_days", "away_rest_days"]):
                result_df["ft_rest_advantage"] = df["home_rest_days"] - df["away_rest_days"]

            new_features = len(result_df.columns) - len(df.columns)
            self.logger.info(f"Generated {new_features} interaction features")

            return result_df

        except Exception as e:
            self.logger.error(f"Error generating interaction features: {str(e)}")
            return df

    def run_hybrid_feature_engineering(
        self,
        df: pd.DataFrame,
        include_temporal: bool = True,
        include_relational: bool = True,
        include_interactions: bool = True,
        temporal_window: int = 5,
        temporal_gap: int = 1,
    ) -> tuple[pd.DataFrame, dict[str, list]]:
        """
        Run the complete hybrid feature engineering pipeline.

        Args:
            df: Input dataframe with existing features
            include_temporal: Whether to generate temporal features
            include_relational: Whether to generate relational features
            include_interactions: Whether to generate interaction features
            temporal_window: Window length for temporal features
            temporal_gap: Gap for temporal features

        Returns:
            Tuple of (augmented_dataframe, feature_definitions_dict)
        """
        try:
            # OPTIMIZATION: Memory monitoring and performance tracking
            initial_memory = df.memory_usage(deep=True).sum() / 1024**2

            self.logger.info(
                "Starting optimized hybrid feature engineering pipeline",
                extra={
                    "include_temporal": include_temporal,
                    "include_relational": include_relational,
                    "include_interactions": include_interactions,
                    "temporal_window": temporal_window,
                    "temporal_gap": temporal_gap,
                    "max_depth": self.max_depth,
                    "input_features": len(df.columns),
                    "initial_memory_mb": initial_memory,
                    "chunk_size": self.chunk_size,
                },
            )

            # OPTIMIZATION: Chunking for large datasets
            if len(df) > self.chunk_size:
                self.logger.info(
                    f"Dataset size ({len(df)}) exceeds chunk size ({self.chunk_size}), using chunked processing"
                )
                return self._run_chunked_feature_engineering(
                    df,
                    include_temporal,
                    include_relational,
                    include_interactions,
                    temporal_window,
                    temporal_gap,
                )

            # Step 1: Prepare data and create EntitySet
            entities = self.prepare_data_for_featuretools(df)
            entityset = self.create_entityset(entities)

            result_df = df.copy()
            all_feature_defs = {}

            # Step 2: Generate temporal features using rolling primitives
            if include_temporal:
                try:
                    # Create temporal primitives with soccer-specific parameters
                    temporal_primitives = [
                        RollingMean(window_length=temporal_window, gap=temporal_gap),
                        RollingMax(window_length=temporal_window, gap=temporal_gap),
                        RollingMin(window_length=temporal_window, gap=temporal_gap),
                        Lag(periods=temporal_gap),
                        Lag(periods=temporal_gap + 2),
                    ]

                    # Generate temporal features with warning suppression
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        warnings.filterwarnings("ignore", message="Could not infer format")

                        temporal_fm, temporal_defs = ft.dfs(
                            entityset=entityset,
                            target_dataframe_name="matches",
                            trans_primitives=temporal_primitives,
                            agg_primitives=[],
                            max_depth=1,
                            n_jobs=self.n_jobs,
                            verbose=False,
                        )

                    if not temporal_fm.empty:
                        # Select only new numeric features
                        new_temporal_cols = [
                            col for col in temporal_fm.columns if col not in df.columns
                        ]
                        if new_temporal_cols:
                            temporal_features = temporal_fm[new_temporal_cols].select_dtypes(
                                include=[np.number]
                            )

                            # Add prefix to new features
                            temporal_features.columns = [
                                f"ft_temporal_{col}" for col in temporal_features.columns
                            ]

                            result_df = result_df.join(temporal_features, how="left")
                            all_feature_defs["temporal"] = temporal_defs

                            self.logger.info(
                                f"Added {len(temporal_features.columns)} temporal features"
                            )

                except Exception as e:
                    self.logger.warning(f"Temporal feature generation failed: {str(e)}")
                    self.logger.info(
                        "Skipping temporal features due to error - this is normal for small datasets or insufficient temporal data"
                    )

            # Step 3: Generate relational features
            if include_relational and len(entityset.dataframes) > 1:
                try:
                    # Generate relational features with warning suppression
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        warnings.filterwarnings("ignore", message="Could not infer format")

                        relational_fm, relational_defs = ft.dfs(
                            entityset=entityset,
                            target_dataframe_name="matches",
                            trans_primitives=[],
                            agg_primitives=[Mean, Max, Min, Count],
                            max_depth=self.max_depth,
                            n_jobs=self.n_jobs,
                            verbose=False,
                        )

                    if not relational_fm.empty:
                        # Select only new numeric features
                        new_relational_cols = [
                            col for col in relational_fm.columns if col not in df.columns
                        ]
                        if new_relational_cols:
                            relational_features = relational_fm[new_relational_cols].select_dtypes(
                                include=[np.number]
                            )

                            # Add prefix to new features
                            relational_features.columns = [
                                f"ft_relational_{col}" for col in relational_features.columns
                            ]

                            result_df = result_df.join(relational_features, how="left")
                            all_feature_defs["relational"] = relational_defs

                            self.logger.info(
                                f"Added {len(relational_features.columns)} relational features"
                            )

                except Exception as e:
                    self.logger.warning(f"Relational feature generation failed: {str(e)}")
                    self.logger.info(
                        "Skipping relational features due to error - this is normal for datasets without sufficient entity relationships"
                    )

            # Step 4: Generate interaction features
            if include_interactions:
                result_df = self.generate_interaction_features(result_df)
                all_feature_defs["interactions"] = ["soccer_specific_interactions"]

            # Step 5: Clean up and validate
            result_df = self._clean_generated_features(result_df, df)

            # OPTIMIZATION: Performance and memory reporting
            final_memory = result_df.memory_usage(deep=True).sum() / 1024**2
            new_features = len(result_df.columns) - len(df.columns)
            memory_increase = final_memory - initial_memory

            # Update performance stats
            self._feature_generation_stats.update(
                {
                    "total_new_features": new_features,
                    "memory_usage_mb": final_memory,
                    "memory_increase_mb": memory_increase,
                }
            )

            self.logger.info(
                f"Hybrid feature engineering completed. Added {new_features} new features",
                extra={
                    "new_features": new_features,
                    "final_memory_mb": final_memory,
                    "memory_increase_mb": memory_increase,
                    "features_per_mb": new_features / max(memory_increase, 0.1),
                    "output_features": len(result_df.columns),
                },
            )

            return result_df, all_feature_defs

        except Exception as e:
            self.logger.error(
                f"Error in hybrid feature engineering: {str(e)}",
                extra={"function": "run_hybrid_feature_engineering"},
            )
            raise

    def _clean_generated_features(
        self, result_df: pd.DataFrame, original_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Clean and validate generated features with improved filtering.

        Args:
            result_df: Dataframe with generated features
            original_df: Original input dataframe

        Returns:
            Cleaned dataframe
        """
        try:
            # Handle infinite values
            result_df = result_df.replace([np.inf, -np.inf], np.nan)

            # Fill NaN values with 0 for new features only
            new_columns = [col for col in result_df.columns if col not in original_df.columns]
            result_df[new_columns] = result_df[new_columns].fillna(0)

            # Enhanced feature filtering - be less aggressive
            features_to_remove = []

            for col in new_columns:
                col_data = result_df[col]

                # Remove only truly problematic features
                if col_data.std() == 0 and col_data.nunique() <= 1:
                    # Only remove if completely constant (all same value)
                    features_to_remove.append(col)
                elif col_data.isnull().sum() > len(result_df) * 0.95:
                    # Remove if >95% missing values
                    features_to_remove.append(col)
                elif abs(col_data.max()) < 1e-10 and abs(col_data.min()) < 1e-10:
                    # Remove if all values are essentially zero
                    features_to_remove.append(col)

            if features_to_remove:
                result_df = result_df.drop(columns=features_to_remove)
                self.logger.info(
                    f"Removed {len(features_to_remove)} problematic features (constant/missing/zero)"
                )
            else:
                self.logger.info("No problematic features found - all generated features retained")

            return result_df

        except Exception as e:
            self.logger.error(f"Error cleaning generated features: {str(e)}")
            return result_df

    def _run_chunked_feature_engineering(
        self,
        df: pd.DataFrame,
        include_temporal: bool,
        include_relational: bool,
        include_interactions: bool,
        temporal_window: int,
        temporal_gap: int,
    ) -> tuple[pd.DataFrame, dict[str, list]]:
        """Run feature engineering in chunks for large datasets."""
        try:
            self.logger.info(f"Processing {len(df)} rows in chunks of {self.chunk_size}")

            chunks = [df[i : i + self.chunk_size] for i in range(0, len(df), self.chunk_size)]
            processed_chunks = []
            all_feature_defs = {}

            for i, chunk in enumerate(chunks):
                self.logger.debug(f"Processing chunk {i + 1}/{len(chunks)}")

                # Temporarily disable chunking for recursive call
                original_chunk_size = self.chunk_size
                self.chunk_size = len(chunk) + 1  # Ensure no further chunking

                try:
                    # Process chunk with regular method
                    chunk_result, chunk_defs = self.run_hybrid_feature_engineering(
                        chunk,
                        include_temporal,
                        include_relational,
                        include_interactions,
                        temporal_window,
                        temporal_gap,
                    )
                    processed_chunks.append(chunk_result)

                    # Merge feature definitions
                    for key, value in chunk_defs.items():
                        if key not in all_feature_defs:
                            all_feature_defs[key] = value
                finally:
                    # Restore original chunk size
                    self.chunk_size = original_chunk_size

            # Combine all chunks
            result_df = pd.concat(processed_chunks, ignore_index=True)

            self.logger.info(
                f"Chunked processing completed. Total features: {len(result_df.columns)}"
            )
            return result_df, all_feature_defs

        except Exception as e:
            self.logger.error(f"Error in chunked feature engineering: {str(e)}")
            raise

    def evaluate_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        new_feature_names: list[str],
        method: str = "combined",
        top_k: int = 50,
    ) -> list[str]:
        """
        Optimized feature importance evaluation using multiple methods.

        Args:
            X: Feature matrix
            y: Target variable
            new_feature_names: List of new feature names to evaluate
            method: Evaluation method ('mutual_info', 'correlation', 'variance', 'combined')
            top_k: Number of top features to return

        Returns:
            List of top feature names ranked by importance
        """
        try:
            from sklearn.feature_selection import mutual_info_classif
            from sklearn.preprocessing import StandardScaler

            if len(new_feature_names) == 0:
                return []

            self.logger.info(
                "Starting optimized feature importance evaluation",
                extra={"num_features": len(new_feature_names), "method": method},
            )

            # OPTIMIZATION: Memory-efficient feature selection
            X_new = X[new_feature_names].copy()
            X_new = self._optimize_data_types(X_new)  # Apply data type optimization

            # Remove any remaining NaN/inf values
            X_new = X_new.replace([np.inf, -np.inf], np.nan).fillna(0)

            feature_scores = {}

            if method in ["mutual_info", "combined"]:
                # Mutual information for classification
                try:
                    mi_scores = mutual_info_classif(X_new, y, random_state=42)
                    for i, feature in enumerate(new_feature_names):
                        feature_scores[f"{feature}_mi"] = mi_scores[i]
                    self.logger.info(
                        f"Calculated mutual information scores for {len(new_feature_names)} features"
                    )
                except Exception as e:
                    self.logger.warning(f"Mutual information calculation failed: {str(e)}")

            if method in ["correlation", "combined"]:
                # Correlation with target
                try:
                    for feature in new_feature_names:
                        corr = abs(X_new[feature].corr(y))
                        if not np.isnan(corr):
                            feature_scores[f"{feature}_corr"] = corr
                    self.logger.info(
                        f"Calculated correlation scores for {len(new_feature_names)} features"
                    )
                except Exception as e:
                    self.logger.warning(f"Correlation calculation failed: {str(e)}")

            if method in ["variance", "combined"]:
                # Variance-based scoring (higher variance = more informative)
                try:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_new)
                    for i, feature in enumerate(new_feature_names):
                        variance = np.var(X_scaled[:, i])
                        feature_scores[f"{feature}_var"] = variance
                    self.logger.info(
                        f"Calculated variance scores for {len(new_feature_names)} features"
                    )
                except Exception as e:
                    self.logger.warning(f"Variance calculation failed: {str(e)}")

            if not feature_scores:
                self.logger.warning("No feature scores calculated, returning original list")
                return new_feature_names[:top_k]

            # Combine scores by feature
            combined_scores = {}
            for feature in new_feature_names:
                scores = []
                if f"{feature}_mi" in feature_scores:
                    scores.append(feature_scores[f"{feature}_mi"])
                if f"{feature}_corr" in feature_scores:
                    scores.append(feature_scores[f"{feature}_corr"])
                if f"{feature}_var" in feature_scores:
                    scores.append(feature_scores[f"{feature}_var"])

                if scores:
                    # Use mean of available scores
                    combined_scores[feature] = np.mean(scores)
                else:
                    combined_scores[feature] = 0.0

            # Sort by combined score
            ranked_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            top_features = [feature for feature, score in ranked_features[:top_k]]

            self.logger.info(
                "Feature importance evaluation completed",
                extra={
                    "selected_features": len(top_features),
                    "method": method,
                    "top_feature_score": max(combined_scores.values()) if combined_scores else 0,
                    "avg_feature_score": np.mean(list(combined_scores.values()))
                    if combined_scores
                    else 0,
                },
            )

            # Log top feature scores for debugging
            if ranked_features:
                self.logger.debug(f"Top 5 feature scores: {ranked_features[:5]}")

            return top_features

        except Exception as e:
            self.logger.error(
                f"Error in feature importance evaluation: {str(e)}",
                extra={"function": "evaluate_feature_importance"},
            )
            return new_feature_names[:top_k]

    def get_performance_stats(self) -> dict[str, Union[int, float]]:
        """Get current performance statistics."""
        return self._feature_generation_stats.copy()

    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self._feature_generation_stats = {
            "temporal_features": 0,
            "relational_features": 0,
            "interaction_features": 0,
            "memory_usage_mb": 0,
        }
        self.logger.debug("Performance statistics reset")
