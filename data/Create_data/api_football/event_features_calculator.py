import os

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from pyexcelerate import Workbook


def _preprocess_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the events DataFrame:
    - Sorts events by fixture_id and time_elapsed.
    - Ensures necessary columns exist.
    - Identifies goal events.
    """
    if events_df.empty:
        return pd.DataFrame(columns=['fixture_id', 'time_elapsed', 'event_type', 'event_detail', 'team_id', 'home_team_id', 'away_team_id', 'is_goal'])

    events_df = events_df.sort_values(by=['fixture_id', 'time_elapsed']).copy()
    
    # Ensure essential columns are present
    required_cols = ['fixture_id', 'time_elapsed', 'event_type', 'event_detail', 'team_id', 'home_team_id', 'away_team_id']
    for col in required_cols:
        if col not in events_df.columns:
            # Add missing columns with NaNs or appropriate defaults if known
            if col in ['home_team_id', 'away_team_id', 'team_id']: # These are IDs, float is ok for NaN
                events_df[col] = np.nan
            else:
                events_df[col] = pd.NA
            print(f"Warning: Column '{col}' was missing and has been added with NaNs.")

    events_df['is_goal'] = events_df['event_type'] == 'Goal'
    return events_df

def export_to_xlsx_fast(df, path):
    # Replace NaN/None with empty string
    df = df.fillna('')
    wb = Workbook()
    wb.new_sheet("Sheet1", data=[df.columns.tolist()] + df.values.tolist())
    wb.save(path)

def calculate_event_features_vectorized(events_df: pd.DataFrame, fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates event-based features for each fixture in a vectorized manner.

    Args:
        events_df (pd.DataFrame): DataFrame containing event data. 
                                    Required columns: fixture_id, time_elapsed, event_type, 
                                                    event_detail, team_id, home_team_id, away_team_id.
        fixtures_df (pd.DataFrame): DataFrame containing fixture data.
                                    Required columns: fixture_id, home_team_id, away_team_id.
                                    Used to get home/away team_id for all fixtures, even those with no events.

    Returns:
        pd.DataFrame: DataFrame with fixture_id and calculated event features.
    """
    if events_df.empty:
        print("Warning: events_df is empty. Returning empty features DataFrame.")
        # Create an empty df with expected feature columns if possible, or just fixture_id
        return pd.DataFrame(columns=['fixture_id'] + get_feature_names())


    # Ensure fixtures_df has the necessary columns
    if not all(col in fixtures_df.columns for col in ['fixture_id', 'home_team_id', 'away_team_id']):
        raise ValueError("fixtures_df must contain 'fixture_id', 'home_team_id', and 'away_team_id'.")

    # Merge home/away team_ids from fixtures_df to events_df to ensure they are present and correct
    events_df = pd.merge(events_df.drop(columns=['home_team_id', 'away_team_id'], errors='ignore'), 
                        fixtures_df[['fixture_id', 'home_team_id', 'away_team_id']], 
                        on='fixture_id', 
                        how='left')

    processed_events_df = _preprocess_events(events_df)
    
    if processed_events_df.empty and not events_df.empty: # Preprocessing made it empty, likely due to missing critical columns
        print("Warning: processed_events_df became empty after preprocessing. Check input events_df columns.")
        return pd.DataFrame(columns=['fixture_id'] + get_feature_names())
    elif processed_events_df.empty and events_df.empty: # Was already empty
        return pd.DataFrame(columns=['fixture_id'] + get_feature_names())

    # Initialize features_df with all unique fixture_ids from fixtures_df
    features_df = fixtures_df[['fixture_id', 'home_team_id', 'away_team_id']].copy()
    features_df = features_df.drop_duplicates(subset=['fixture_id']).set_index('fixture_id')

    # --- Feature Calculation Functions Will Be Called Here ---
    feature_functions_map = {
        'is_score_tied_at_80_min': _calculate_is_score_tied_at_80_min_vectorized,
        'abs_goal_difference_at_45_min': _calculate_abs_goal_difference_at_45_min_vectorized,
        'late_equalizer_scored_in_last_15_mins': _calculate_late_equalizer_scored_in_last_15_mins_vectorized,
        'red_card_to_leading_team_after_60_min': _calculate_red_card_to_leading_team_after_60_min_vectorized,
        'max_continuous_time_score_tied_overall': _calculate_max_continuous_time_score_tied_overall_vectorized,
        'low_total_goals_match': _calculate_low_total_goals_match_vectorized,
        'first_goal_scored_late_or_no_goals': _calculate_first_goal_scored_late_or_no_goals_vectorized,
        'late_substitutions_intensity_when_close': _calculate_late_substitutions_intensity_when_close_vectorized,
        'rolling_avg_goal_difference_last_30_min_at_end': _calculate_rolling_avg_goal_difference_last_30_min_at_end_vectorized,
        'mean_time_between_goals_overall': _calculate_mean_time_between_goals_overall_vectorized,
        'abs_shots_on_target_diff_last_20_min': _calculate_abs_shots_on_target_diff_last_20_min_vectorized,
        'total_yellow_cards_first_half': _calculate_total_yellow_cards_first_half_vectorized,
        'abs_goal_diff_0_30_min': _calculate_abs_goal_diff_0_30_min_vectorized,
        'abs_goal_diff_30_60_min': _calculate_abs_goal_diff_30_60_min_vectorized,
        'abs_goal_diff_60_90_min': _calculate_abs_goal_diff_60_90_min_vectorized,
        'abs_substitutions_diff_overall': _calculate_abs_substitutions_diff_overall_vectorized
        # 'proportion_of_game_time_score_is_tied_at_end': _calculate_proportion_of_game_time_score_is_tied_at_end_vectorized
    }

    for feature_name, calc_function in feature_functions_map.items():
        try:
            print(f"Calculating {feature_name}...")
            # Pass fixtures_df as the second argument, which corresponds to fixtures_info_df in helpers
            feature_series = calc_function(processed_events_df, fixtures_df) 
            features_df = features_df.join(feature_series) 
        except Exception as e:
            print(f"Error calculating {feature_name}: {e}. Filling with NaNs.")
            # Create a default series with NaNs, indexed like features_df (which is fixture_id)
            default_series = pd.Series(np.nan, index=features_df.index, name=feature_name)
            features_df = features_df.join(default_series)
    
    # Reset index to make fixture_id a column again
    features_df.reset_index(inplace=True)
    
    # Ensure all defined feature columns are present, even if no events led to their calculation
    expected_cols = ['fixture_id'] + get_feature_names()
    for col in expected_cols:
        if col not in features_df.columns:
            # Default for many features will be 0 if they are counts/flags, or NaN if ratios/means that couldn't be calculated
            if col.startswith('is_') or col.startswith('late_equalizer') or col.startswith('red_card') or col.startswith('low_total') or col.startswith('first_goal_'):
                features_df[col] = 0 
            elif col.startswith('abs_') or col.startswith('total_') or col.startswith('late_subs'):
                features_df[col] = 0
            else: # Means, durations, proportions
                features_df[col] = np.nan
                
    return features_df # Return only expected columns in defined order

def get_feature_names():
    """Returns a list of all feature names that will be calculated."""
    return [
        'is_score_tied_at_80_min',
        'abs_goal_difference_at_45_min',
        'late_equalizer_scored_in_last_15_mins',
        'red_card_to_leading_team_after_60_min',
        'max_continuous_time_score_tied_overall',
        'low_total_goals_match',
        'first_goal_scored_late_or_no_goals',
        'late_substitutions_intensity_when_close',
        'rolling_avg_goal_difference_last_30_min_at_end',
        'mean_time_between_goals_overall',
        'abs_shots_on_target_diff_last_20_min',
        'total_yellow_cards_first_half',
        'abs_goal_diff_0_30_min',
        'abs_goal_diff_30_60_min',
        'abs_goal_diff_60_90_min',
        'abs_substitutions_diff_overall'
        # 'proportion_of_game_time_score_is_tied_at_end'
    ]

# --- Individual Feature Calculation Functions Start Here ---

def _get_score_at_time_vectorized(processed_events_df: pd.DataFrame, time_limit: int, fixtures_info_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates home and away goals at a specific time_limit for all fixtures.

    Args:
        processed_events_df: DataFrame from _preprocess_events. Must include home_team_id, away_team_id.
        time_limit: The time in minutes to get the score at.
        fixtures_info_df: DataFrame with fixture_id, home_team_id, away_team_id for all relevant fixtures.
                            Used to ensure all fixtures are present in the output, even if they had no goals.
    Returns:
        pd.DataFrame: Indexed by fixture_id, with columns 'home_goals_at_time' and 'away_goals_at_time'.
    """
    if processed_events_df.empty:
        # Create a DataFrame with all fixtures and 0 goals if no events
        scores_at_time = fixtures_info_df[['fixture_id']].copy()
        scores_at_time['home_goals_at_time'] = 0
        scores_at_time['away_goals_at_time'] = 0
        return scores_at_time.set_index('fixture_id')

    events_at_time_limit = processed_events_df[processed_events_df['time_elapsed'] <= time_limit].copy()

    if events_at_time_limit.empty: # No events at or before the time limit for any fixture
        scores_at_time = fixtures_info_df[['fixture_id']].copy()
        scores_at_time['home_goals_at_time'] = 0
        scores_at_time['away_goals_at_time'] = 0
        return scores_at_time.set_index('fixture_id')

    # Ensure 'is_goal' is boolean for sum()
    events_at_time_limit['is_goal'] = events_at_time_limit['is_goal'].astype(bool)
    
    # Calculate goals for home team and away team
    events_at_time_limit['is_home_goal'] = (events_at_time_limit['is_goal']) & (events_at_time_limit['team_id'] == events_at_time_limit['home_team_id'])
    events_at_time_limit['is_away_goal'] = (events_at_time_limit['is_goal']) & (events_at_time_limit['team_id'] == events_at_time_limit['away_team_id'])

    scores_at_time = events_at_time_limit.groupby('fixture_id').agg(
        home_goals_at_time=('is_home_goal', 'sum'),
        away_goals_at_time=('is_away_goal', 'sum')
    ).reset_index()
    
    # Merge with fixtures_info_df to include all fixtures, defaulting missing ones to 0 goals
    # Need home_team_id, away_team_id from fixtures_info_df for context
    all_fixtures_scores = pd.merge(
        fixtures_info_df[['fixture_id', 'home_team_id', 'away_team_id']], 
        scores_at_time, 
        on='fixture_id', 
        how='left'
    ).fillna({'home_goals_at_time': 0, 'away_goals_at_time': 0})
    
    all_fixtures_scores['home_goals_at_time'] = all_fixtures_scores['home_goals_at_time'].astype(int)
    all_fixtures_scores['away_goals_at_time'] = all_fixtures_scores['away_goals_at_time'].astype(int)
    
    return all_fixtures_scores.set_index('fixture_id')[['home_goals_at_time', 'away_goals_at_time']]

def _calculate_is_score_tied_at_80_min_vectorized(processed_events_df: pd.DataFrame, fixtures_info_df: pd.DataFrame) -> pd.Series:
    """
    Calculates if the score was tied at the 80th minute for each fixture.
    Args:
        processed_events_df: DataFrame from _preprocess_events, must include home_team_id, away_team_id.
        fixtures_info_df: DataFrame with fixture_id, home_team_id, away_team_id for all relevant fixtures.
    Returns:
        pd.Series: Indexed by fixture_id, value is 1 if tied at 80th min, 0 otherwise.
                    Fixtures with no goals or events before 80 min are considered tied 0-0.
    """
    scores_at_80_min = _get_score_at_time_vectorized(processed_events_df, 80, fixtures_info_df)
    
    is_tied_series = (scores_at_80_min['home_goals_at_time'] == scores_at_80_min['away_goals_at_time']).astype(int)
    is_tied_series.name = 'is_score_tied_at_80_min'
    return is_tied_series

def _calculate_abs_goal_difference_at_45_min_vectorized(processed_events_df: pd.DataFrame, fixtures_info_df: pd.DataFrame) -> pd.Series:
    """
    Calculates the absolute goal difference at the 45th minute for each fixture.
    Args:
        processed_events_df: DataFrame from _preprocess_events.
        fixtures_info_df: DataFrame with fixture_id, home_team_id, away_team_id.
    Returns:
        pd.Series: Indexed by fixture_id, value is the absolute goal difference at 45th min.
                    Fixtures with no goals by 45th min have a difference of 0.
    """
    scores_at_45_min = _get_score_at_time_vectorized(processed_events_df, 45, fixtures_info_df)
    
    abs_diff_series = (scores_at_45_min['home_goals_at_time'] - scores_at_45_min['away_goals_at_time']).abs().astype(int)
    abs_diff_series.name = 'abs_goal_difference_at_45_min'
    return abs_diff_series

def _late_equalizer_check_fixture(fixture_events: pd.DataFrame) -> int:
    """
    Helper function for a single fixture to check for a late equalizer.
    Assumes fixture_events are sorted by time_elapsed and contain home_team_id, away_team_id.
    """
    if fixture_events.empty:
        return 0

    # Ensure home_team_id and away_team_id are consistent for the fixture
    home_team_id = fixture_events['home_team_id'].iloc[0]
    away_team_id = fixture_events['away_team_id'].iloc[0]

    late_goals = fixture_events[
        (fixture_events['is_goal']) &
        (fixture_events['time_elapsed'] >= 75)
    ].copy()

    if late_goals.empty:
        return 0

    # Calculate cumulative scores *within this fixture* up to each event time
    # This is a simplified version of _get_score_at_time for a single fixture context
    fixture_events_sorted = fixture_events.sort_values(by='time_elapsed')
    fixture_events_sorted['is_home_goal_event'] = (fixture_events_sorted['is_goal']) & (fixture_events_sorted['team_id'] == home_team_id)
    fixture_events_sorted['is_away_goal_event'] = (fixture_events_sorted['is_goal']) & (fixture_events_sorted['team_id'] == away_team_id)
    
    fixture_events_sorted['current_home_score'] = fixture_events_sorted['is_home_goal_event'].cumsum()
    fixture_events_sorted['current_away_score'] = fixture_events_sorted['is_away_goal_event'].cumsum()

    for _, goal_event in late_goals.iterrows():
        time_of_goal = goal_event['time_elapsed']

        # Score just BEFORE this specific late goal (using events strictly before this one)
        events_before_goal = fixture_events_sorted[fixture_events_sorted['time_elapsed'] < time_of_goal]
        if not events_before_goal.empty:
            home_goals_before_event = events_before_goal.iloc[-1]['current_home_score']
            away_goals_before_event = events_before_goal.iloc[-1]['current_away_score']
        else: # This goal is the first event, or first goal
            home_goals_before_event = 0
            away_goals_before_event = 0
        
        # Score right AFTER (i.e. including) this specific late goal
        # We need to find the state of current_home_score and current_away_score for *this* goal_event row
        # from fixture_events_sorted, which has the cumulative scores including this goal.
        current_event_scores = fixture_events_sorted[fixture_events_sorted.index == goal_event.name]
        home_goals_after_event = current_event_scores['current_home_score'].iloc[0]
        away_goals_after_event = current_event_scores['current_away_score'].iloc[0]

        if home_goals_after_event == away_goals_after_event and home_goals_before_event != away_goals_before_event:
            return 1
    return 0

def _calculate_late_equalizer_scored_in_last_15_mins_vectorized(processed_events_df: pd.DataFrame, fixtures_info_df: pd.DataFrame) -> pd.Series:
    """
    Calculates if a late equalizer was scored (from 75th min onwards) for each fixture.
    Args:
        processed_events_df: DataFrame from _preprocess_events, must include home_team_id, away_team_id.
        fixtures_info_df: DataFrame with fixture_id, home_team_id, away_team_id.
    Returns:
        pd.Series: Indexed by fixture_id, value is 1 if late equalizer scored, 0 otherwise.
                    Defaults to 0 for fixtures with no late goals or no events.
    """
    if processed_events_df.empty:
        result_series = pd.Series(0, index=fixtures_info_df['fixture_id'].unique(), name='late_equalizer_scored_in_last_15_mins')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(0)

    # Ensure home_team_id and away_team_id are present from fixtures_info_df merge in main func.
    # These columns must be in processed_events_df.
    if not all(col in processed_events_df.columns for col in ['home_team_id', 'away_team_id']):
        raise ValueError("processed_events_df must contain 'home_team_id' and 'away_team_id' for late equalizer check.")

    # Apply the helper function to each fixture's events
    # The `fixture_events` passed to `_late_equalizer_check_fixture` will already have the correct home/away team_id
    # due to the merge in `calculate_event_features_vectorized`.
    late_equalizer_series = processed_events_df.groupby('fixture_id').apply(_late_equalizer_check_fixture, include_groups=False)
    late_equalizer_series.name = 'late_equalizer_scored_in_last_15_mins'

    # Ensure all fixtures from fixtures_info_df are present, defaulting to 0
    # Create a base series from fixtures_info_df to ensure all fixture_ids are included
    base_fixture_ids = fixtures_info_df.set_index('fixture_id').index
    result_series = late_equalizer_series.reindex(base_fixture_ids).fillna(0).astype(int)
    
    return result_series

def _red_card_leading_team_check_fixture(fixture_events: pd.DataFrame) -> int:
    """
    Helper for a single fixture to check for a red card to the leading team after 60 mins.
    Assumes fixture_events are sorted and contain home_team_id, away_team_id.
    """
    if fixture_events.empty:
        return 0

    home_team_id = fixture_events['home_team_id'].iloc[0]
    away_team_id = fixture_events['away_team_id'].iloc[0]

    red_card_events_late = fixture_events[
        (fixture_events['event_detail'] == 'Red Card') & # Assuming this is how red cards are marked
        (fixture_events['time_elapsed'] >= 60)
    ].copy()

    if red_card_events_late.empty:
        return 0

    # Calculate cumulative scores within this fixture up to each event time
    fixture_events_sorted = fixture_events.sort_values(by='time_elapsed')
    fixture_events_sorted['is_home_goal_event'] = (fixture_events_sorted['is_goal']) & (fixture_events_sorted['team_id'] == home_team_id)
    fixture_events_sorted['is_away_goal_event'] = (fixture_events_sorted['is_goal']) & (fixture_events_sorted['team_id'] == away_team_id)
    fixture_events_sorted['current_home_score'] = fixture_events_sorted['is_home_goal_event'].cumsum()
    fixture_events_sorted['current_away_score'] = fixture_events_sorted['is_away_goal_event'].cumsum()

    for _, card_event in red_card_events_late.iterrows():
        time_of_card = card_event['time_elapsed']
        carded_team_id = card_event['team_id']

        events_before_card = fixture_events_sorted[fixture_events_sorted['time_elapsed'] < time_of_card]
        if not events_before_card.empty:
            home_goals_before_card = events_before_card.iloc[-1]['current_home_score']
            away_goals_before_card = events_before_card.iloc[-1]['current_away_score']
        else: # Card is the first event, or no goals before it
            home_goals_before_card = 0
            away_goals_before_card = 0
        
        if home_goals_before_card > away_goals_before_card and carded_team_id == home_team_id:
            return 1 # Home team was leading and got a red card
        if away_goals_before_card > home_goals_before_card and carded_team_id == away_team_id:
            return 1 # Away team was leading and got a red card
            
    return 0

def _calculate_red_card_to_leading_team_after_60_min_vectorized(processed_events_df: pd.DataFrame, fixtures_info_df: pd.DataFrame) -> pd.Series:
    """
    Calculates if a leading team received a red card after the 60th minute.
    Args:
        processed_events_df: DataFrame from _preprocess_events.
        fixtures_info_df: DataFrame with fixture_id, home_team_id, away_team_id.
    Returns:
        pd.Series: Indexed by fixture_id, value is 1 if condition met, 0 otherwise.
    """
    if processed_events_df.empty:
        result_series = pd.Series(0, index=fixtures_info_df['fixture_id'].unique(), name='red_card_to_leading_team_after_60_min')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(0)

    if not all(col in processed_events_df.columns for col in ['home_team_id', 'away_team_id', 'event_detail']):
        # If event_detail is missing, this feature cannot be calculated as specified.
        print("Warning: 'home_team_id', 'away_team_id', or 'event_detail' missing in processed_events_df for red card check. Returning 0s.")
        result_series = pd.Series(0, index=fixtures_info_df['fixture_id'].unique(), name='red_card_to_leading_team_after_60_min')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(0).astype(int)

    red_card_series = processed_events_df.groupby('fixture_id').apply(_red_card_leading_team_check_fixture, include_groups=False)
    red_card_series.name = 'red_card_to_leading_team_after_60_min'
    
    base_fixture_ids = fixtures_info_df.set_index('fixture_id').index
    result_series = red_card_series.reindex(base_fixture_ids).fillna(0).astype(int)
    return result_series

def _max_continuous_time_tied_fixture(fixture_events: pd.DataFrame) -> float:
    """
    Helper for a single fixture to calculate max continuous time score was tied.
    Assumes fixture_events are sorted and contain home_team_id, away_team_id.
    """
    if fixture_events.empty: # No events, score is 0-0 for the whole match (assuming 90 min)
        return 90.0 

    home_team_id = fixture_events['home_team_id'].iloc[0]
    away_team_id = fixture_events['away_team_id'].iloc[0]

    # Relevant events are goals. Also consider match start and end.
    goal_events_times = fixture_events[fixture_events['is_goal']]['time_elapsed'].unique()
    
    # Define critical time points: start, goal times, and match end.
    # Assume match end is 90, or max event time if events go beyond that (e.g. injury time goals)
    match_end_time = 90.0
    if not fixture_events.empty:
        max_event_time = fixture_events['time_elapsed'].max()
        if max_event_time > match_end_time: # Handle cases where events (like goals) are recorded past 90 min
            match_end_time = max_event_time
            
    time_points = sorted(list(set([0.0] + list(goal_events_times) + [match_end_time])))
    # Filter time_points to be within 0 and match_end_time, and ensure they are unique and sorted.
    time_points = sorted(list(set(tp for tp in time_points if 0 <= tp <= match_end_time)))
    if not time_points or time_points[-1] < match_end_time:
         if match_end_time not in time_points:
            time_points.append(match_end_time)
         time_points = sorted(list(set(time_points)))
    if 0.0 not in time_points:
        time_points.insert(0, 0.0)

    max_tied_duration = 0.0
    current_tied_streak_start_time = 0.0
    score_is_currently_tied = True # Starts 0-0 at time 0.0

    # Pre-calculate scores at each event to avoid repeated full scans for a single fixture
    # This part is crucial for performance within the apply function.
    # We need running scores at each of the *original* event times.
    _fixture_events_sorted = fixture_events.sort_values(by='time_elapsed').copy()
    _fixture_events_sorted['is_home_g'] = (_fixture_events_sorted['is_goal']) & (_fixture_events_sorted['team_id'] == home_team_id)
    _fixture_events_sorted['is_away_g'] = (_fixture_events_sorted['is_goal']) & (_fixture_events_sorted['team_id'] == away_team_id)
    _fixture_events_sorted['cum_home_g'] = _fixture_events_sorted['is_home_g'].cumsum()
    _fixture_events_sorted['cum_away_g'] = _fixture_events_sorted['is_away_g'].cumsum()

    # Create a lookup for scores at any given time for this fixture
    def get_score_at_t(t_limit, _events_with_cum_scores):
        relevant_events = _events_with_cum_scores[_events_with_cum_scores['time_elapsed'] <= t_limit]
        if relevant_events.empty:
            return 0, 0
        # Last event at or before t_limit holds the cumulative score
        last_event_at_t = relevant_events.iloc[-1]
        return last_event_at_t['cum_home_g'], last_event_at_t['cum_away_g']

    for i in range(len(time_points)):
        current_segment_end_time = time_points[i]
        
        # Score at the beginning of the current segment (end of last segment)
        # For the first segment (i=0, current_segment_end_time = 0.0), this is 0-0
        # For subsequent segments, it's the score at time_points[i-1]
        prev_segment_end_time = time_points[i-1] if i > 0 else 0.0
        
        # We need score *at* current_segment_end_time to decide if the state *changes* at this point.
        # But the duration calculation depends on the state *during* the segment from prev to current.

        # If the score was tied leading into this segment (or at the start of this segment)
        if score_is_currently_tied:
            # This segment (from prev_segment_end_time to current_segment_end_time) was a tied period.
            # Add its duration to max_tied_duration if it's the end of the streak.
            pass # Covered by logic below
        
        # Determine score at current_segment_end_time
        h_goals_at_current_t, a_goals_at_current_t = get_score_at_t(current_segment_end_time, _fixture_events_sorted)

        if score_is_currently_tied:
            if h_goals_at_current_t != a_goals_at_current_t: # Score just became un-tied at current_segment_end_time
                duration = current_segment_end_time - current_tied_streak_start_time
                max_tied_duration = max(max_tied_duration, duration)
                score_is_currently_tied = False
            # If it's still tied, the streak continues. current_tied_streak_start_time remains unchanged.
        else: # Score was not tied leading into this point
            if h_goals_at_current_t == a_goals_at_current_t: # Score just became tied at current_segment_end_time
                score_is_currently_tied = True
                current_tied_streak_start_time = current_segment_end_time # New tied streak starts now
        
        # Special handling for the very last segment if the match ends tied
        if i == len(time_points) - 1 and score_is_currently_tied:
            # If it's the last point (match_end_time) and score is still tied
            duration = match_end_time - current_tied_streak_start_time
            max_tied_duration = max(max_tied_duration, duration)
            
    return round(max_tied_duration, 2)

def _calculate_max_continuous_time_score_tied_overall_vectorized(processed_events_df: pd.DataFrame, fixtures_info_df: pd.DataFrame) -> pd.Series:
    """
    Calculates the maximum continuous time the score was tied in minutes.
    Args:
        processed_events_df: DataFrame from _preprocess_events.
        fixtures_info_df: DataFrame with fixture_id, home_team_id, away_team_id.
    Returns:
        pd.Series: Indexed by fixture_id, value is the max continuous tied time (float).
                    Defaults to 90.0 for fixtures with no events (assumed 0-0 for full match).
                    Defaults to 0.0 if processing for a fixture with events leads to an issue (e.g. no home/away id, though guarded).
    """
    if processed_events_df.empty:
        # Assume 90 mins tied if no events (0-0 for whole match)
        result_series = pd.Series(90.0, index=fixtures_info_df['fixture_id'].unique(), name='max_continuous_time_score_tied_overall')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(90.0)

    if not all(col in processed_events_df.columns for col in ['home_team_id', 'away_team_id']):
        # This shouldn't happen if pre-processing and merges are correct
        print("Warning: 'home_team_id' or 'away_team_id' missing for max_continuous_time_score_tied. Returning 0s.")
        result_series = pd.Series(0.0, index=fixtures_info_df['fixture_id'].unique(), name='max_continuous_time_score_tied_overall')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(0.0)

    max_tied_time_series = processed_events_df.groupby('fixture_id').apply(_max_continuous_time_tied_fixture, include_groups=False)
    max_tied_time_series.name = 'max_continuous_time_score_tied_overall'
    
    base_fixture_ids = fixtures_info_df.set_index('fixture_id').index
    result_series = max_tied_time_series.reindex(base_fixture_ids).fillna(90.0) # Default for missing fixtures (no events)
    
    return result_series

def _calculate_low_total_goals_match_vectorized(processed_events_df: pd.DataFrame, fixtures_info_df: pd.DataFrame) -> pd.Series:
    """
    Calculates if the match had low total goals (0 or 2).
    Args:
        processed_events_df: DataFrame from _preprocess_events.
        fixtures_info_df: DataFrame with fixture_id for all relevant fixtures.
    Returns:
        pd.Series: Indexed by fixture_id, value is 1 if 0 or 2 total goals, 0 otherwise.
                   Defaults to 1 for fixtures with no events (0 goals).
    """
    if processed_events_df.empty:
        # No events means 0 goals, which is a low total goals match.
        result_series = pd.Series(1, index=fixtures_info_df['fixture_id'].unique(), name='low_total_goals_match')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(1).astype(int)

    # Count total goals per fixture
    total_goals_per_fixture = processed_events_df.groupby('fixture_id')['is_goal'].sum()
    total_goals_per_fixture.name = 'total_goals'

    # Check condition (0 or 2 goals)
    low_goals_series = ((total_goals_per_fixture == 0) | (total_goals_per_fixture == 2)).astype(int)
    low_goals_series.name = 'low_total_goals_match'

    # Ensure all fixtures are present, defaulting appropriately
    # Fixtures not in total_goals_per_fixture had 0 goals, so they satisfy the condition (0 goals).
    base_fixture_ids = fixtures_info_df.set_index('fixture_id').index
    result_series = low_goals_series.reindex(base_fixture_ids).fillna(1).astype(int) # Default to 1 (0 goals)
    
    return result_series

def _calculate_first_goal_scored_late_or_no_goals_vectorized(processed_events_df: pd.DataFrame, fixtures_info_df: pd.DataFrame) -> pd.Series:
    """
    Calculates if the first goal was scored late (>=70 min) or if there were no goals.
    Args:
        processed_events_df: DataFrame from _preprocess_events.
        fixtures_info_df: DataFrame with fixture_id for all relevant fixtures.
    Returns:
        pd.Series: Indexed by fixture_id, value is 1 if condition met, 0 otherwise.
                   Defaults to 1 for fixtures with no events (no goals).
    """
    if processed_events_df.empty:
        # No events means no goals, so condition is met.
        result_series = pd.Series(1, index=fixtures_info_df['fixture_id'].unique(), name='first_goal_scored_late_or_no_goals')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(1).astype(int)

    goal_events = processed_events_df[processed_events_df['is_goal']].copy()

    if goal_events.empty:
        # No goal events across all fixtures, so all meet the 'no goals' condition.
        result_series = pd.Series(1, index=fixtures_info_df['fixture_id'].unique(), name='first_goal_scored_late_or_no_goals')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(1).astype(int)

    # Time of the first goal for each fixture that has goals
    first_goal_times = goal_events.groupby('fixture_id')['time_elapsed'].min()
    first_goal_times.name = 'first_goal_time'

    # Check condition: first goal time >= 70
    late_first_goal_series = (first_goal_times >= 70).astype(int)
    late_first_goal_series.name = 'first_goal_scored_late_or_no_goals'

    # Merge with all fixtures. Fixtures not in late_first_goal_series either had no goals (condition met)
    # or their first goal was < 70 (condition not met, but fillna(1) handles the no-goal case correctly).
    base_fixture_ids = fixtures_info_df.set_index('fixture_id').index
    result_series = late_first_goal_series.reindex(base_fixture_ids).fillna(1).astype(int) # Default to 1 (no goals implies condition met)
    
    return result_series

def _late_subs_intensity_check_fixture(fixture_events: pd.DataFrame) -> int:
    """
    Helper for a single fixture to count late substitutions when score is close.
    Assumes fixture_events are sorted and have home_team_id, away_team_id.
    """
    if fixture_events.empty:
        return 0

    home_team_id = fixture_events['home_team_id'].iloc[0]
    away_team_id = fixture_events['away_team_id'].iloc[0]

    # Assuming 'subst' is the event_type for substitution
    late_subs = fixture_events[
        (fixture_events['event_type'] == 'subst') &
        (fixture_events['time_elapsed'] >= 70)
    ].copy()

    if late_subs.empty:
        return 0

    # Pre-calculate cumulative scores for this fixture
    _fixture_events_sorted = fixture_events.sort_values(by='time_elapsed').copy()
    _fixture_events_sorted['is_home_g'] = (_fixture_events_sorted['is_goal']) & (_fixture_events_sorted['team_id'] == home_team_id)
    _fixture_events_sorted['is_away_g'] = (_fixture_events_sorted['is_goal']) & (_fixture_events_sorted['team_id'] == away_team_id)
    _fixture_events_sorted['cum_home_g'] = _fixture_events_sorted['is_home_g'].cumsum()
    _fixture_events_sorted['cum_away_g'] = _fixture_events_sorted['is_away_g'].cumsum()

    def get_score_just_before_t(t_event, _events_with_cum_scores):
        # Events strictly before t_event
        relevant_events = _events_with_cum_scores[_events_with_cum_scores['time_elapsed'] < t_event]
        if relevant_events.empty:
            return 0, 0
        last_event_before_t = relevant_events.iloc[-1]
        return last_event_before_t['cum_home_g'], last_event_before_t['cum_away_g']

    intensity_count = 0
    for _, sub_event in late_subs.iterrows():
        time_of_sub = sub_event['time_elapsed']
        
        home_goals_before_sub, away_goals_before_sub = get_score_just_before_t(time_of_sub, _fixture_events_sorted)
        
        if abs(home_goals_before_sub - away_goals_before_sub) <= 1:
            intensity_count += 1
            
    return intensity_count

def _calculate_late_substitutions_intensity_when_close_vectorized(processed_events_df: pd.DataFrame, fixtures_info_df: pd.DataFrame) -> pd.Series:
    """
    Calculates intensity of late substitutions when the score is close.
    Args:
        processed_events_df: DataFrame from _preprocess_events.
        fixtures_info_df: DataFrame with fixture_id, home_team_id, away_team_id.
    Returns:
        pd.Series: Indexed by fixture_id, value is count of such substitutions.
                    Defaults to 0 for fixtures with no relevant events.
    """
    if processed_events_df.empty or 'subst' not in processed_events_df['event_type'].unique():
        result_series = pd.Series(0, index=fixtures_info_df['fixture_id'].unique(), name='late_substitutions_intensity_when_close')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(0).astype(int)

    if not all(col in processed_events_df.columns for col in ['home_team_id', 'away_team_id', 'event_type']):
        print("Warning: Essential columns missing for late_substitutions_intensity. Returning 0s.")
        result_series = pd.Series(0, index=fixtures_info_df['fixture_id'].unique(), name='late_substitutions_intensity_when_close')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(0).astype(int)

    intensity_series = processed_events_df.groupby('fixture_id').apply(_late_subs_intensity_check_fixture, include_groups=False)
    intensity_series.name = 'late_substitutions_intensity_when_close'
    
    base_fixture_ids = fixtures_info_df.set_index('fixture_id').index
    result_series = intensity_series.reindex(base_fixture_ids).fillna(0).astype(int)
    return result_series

def _rolling_avg_goal_diff_last_30_fixture(fixture_events: pd.DataFrame) -> float:
    """
    Helper for a single fixture to calculate rolling avg goal diff in last 30 mins (61-90).
    """
    if fixture_events.empty:
        return 0.0 # No events, assume 0-0, so goal diff is 0

    home_team_id = fixture_events['home_team_id'].iloc[0]
    away_team_id = fixture_events['away_team_id'].iloc[0]

    # Pre-calculate cumulative scores for this fixture
    _fixture_events_sorted = fixture_events.sort_values(by='time_elapsed').copy()
    _fixture_events_sorted['is_home_g'] = (_fixture_events_sorted['is_goal']) & (_fixture_events_sorted['team_id'] == home_team_id)
    _fixture_events_sorted['is_away_g'] = (_fixture_events_sorted['is_goal']) & (_fixture_events_sorted['team_id'] == away_team_id)
    _fixture_events_sorted['cum_home_g'] = _fixture_events_sorted['is_home_g'].cumsum()
    _fixture_events_sorted['cum_away_g'] = _fixture_events_sorted['is_away_g'].cumsum()

    def get_score_at_minute_t(minute_val, _events_with_cum_scores):
        relevant_events = _events_with_cum_scores[_events_with_cum_scores['time_elapsed'] <= minute_val]
        if relevant_events.empty:
            return 0, 0
        last_event_at_t = relevant_events.iloc[-1]
        return last_event_at_t['cum_home_g'], last_event_at_t['cum_away_g']

    goal_differences = []
    start_minute = 60
    end_minute = 90 
    # Original logic used range(start_minute + 1, end_minute + 1) -> 61 to 90.

    for minute in range(start_minute + 1, end_minute + 1): # 61, 62, ..., 90
        h_goals, a_goals = get_score_at_minute_t(minute, _fixture_events_sorted)
        goal_differences.append(h_goals - a_goals)
    
    return round(np.mean(goal_differences), 2) if goal_differences else 0.0

def _calculate_rolling_avg_goal_difference_last_30_min_at_end_vectorized(processed_events_df: pd.DataFrame, fixtures_info_df: pd.DataFrame) -> pd.Series:
    """
    Calculates rolling average goal difference in the last 30 minutes (61-90 min).
    Args:
        processed_events_df: DataFrame from _preprocess_events.
        fixtures_info_df: DataFrame with fixture_id, home_team_id, away_team_id.
    Returns:
        pd.Series: Indexed by fixture_id, value is the average goal difference.
                    Defaults to 0.0 for fixtures with no events.
    """
    if processed_events_df.empty:
        result_series = pd.Series(0.0, index=fixtures_info_df['fixture_id'].unique(), name='rolling_avg_goal_difference_last_30_min_at_end')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(0.0)

    if not all(col in processed_events_df.columns for col in ['home_team_id', 'away_team_id']):
        print("Warning: Essential columns missing for rolling_avg_goal_difference. Returning 0.0s.")
        result_series = pd.Series(0.0, index=fixtures_info_df['fixture_id'].unique(), name='rolling_avg_goal_difference_last_30_min_at_end')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(0.0)

    avg_goal_diff_series = processed_events_df.groupby('fixture_id').apply(_rolling_avg_goal_diff_last_30_fixture, include_groups=False)
    avg_goal_diff_series.name = 'rolling_avg_goal_difference_last_30_min_at_end'
    
    base_fixture_ids = fixtures_info_df.set_index('fixture_id').index
    result_series = avg_goal_diff_series.reindex(base_fixture_ids).fillna(0.0) # Default for no events is 0.0 diff
    return result_series

def _mean_time_between_goals_fixture(fixture_goal_events: pd.DataFrame) -> float:
    """
    Helper for a single fixture to calculate mean time between goals.
    Assumes fixture_goal_events contains only goal events for that fixture, sorted by time.
    """
    if fixture_goal_events.shape[0] < 2:
        return np.nan # Not enough goals for a duration "between"
    
    goal_times = fixture_goal_events['time_elapsed'].to_list()
    inter_goal_durations = [goal_times[i] - goal_times[i-1] for i in range(1, len(goal_times))]
    
    return round(np.mean(inter_goal_durations), 2) if inter_goal_durations else np.nan

def _calculate_mean_time_between_goals_overall_vectorized(processed_events_df: pd.DataFrame, fixtures_info_df: pd.DataFrame) -> pd.Series:
    """
    Calculates the mean time between goals for each fixture.
    Args:
        processed_events_df: DataFrame from _preprocess_events.
        fixtures_info_df: DataFrame with fixture_id for all relevant fixtures.
    Returns:
        pd.Series: Indexed by fixture_id, value is the mean time (float) or NaN.
                    Defaults to NaN for fixtures with < 2 goals.
    """
    if processed_events_df.empty:
        result_series = pd.Series(np.nan, index=fixtures_info_df['fixture_id'].unique(), name='mean_time_between_goals_overall')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(np.nan)

    goal_events = processed_events_df[processed_events_df['is_goal']].copy()
    if goal_events.empty:
        result_series = pd.Series(np.nan, index=fixtures_info_df['fixture_id'].unique(), name='mean_time_between_goals_overall')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(np.nan)

    # Sort goal events by fixture and time before applying
    goal_events_sorted = goal_events.sort_values(by=['fixture_id', 'time_elapsed'])
    
    mean_time_series = goal_events_sorted.groupby('fixture_id').apply(_mean_time_between_goals_fixture, include_groups=False)
    mean_time_series.name = 'mean_time_between_goals_overall'
    
    base_fixture_ids = fixtures_info_df.set_index('fixture_id').index
    result_series = mean_time_series.reindex(base_fixture_ids).fillna(np.nan) # Default for no/few goals is NaN
    return result_series

def _calculate_abs_shots_on_target_diff_last_20_min_vectorized(processed_events_df: pd.DataFrame, fixtures_info_df: pd.DataFrame) -> pd.Series:
    """
    Calculates absolute difference in shots on target in the last 20 mins (>=70 min).
    Args:
        processed_events_df: DataFrame from _preprocess_events.
        fixtures_info_df: DataFrame with fixture_id, home_team_id, away_team_id.
    Returns:
        pd.Series: Indexed by fixture_id, value is the absolute difference (int).
                    Defaults to 0 for fixtures with no late SOTs.
    """
    if processed_events_df.empty:
        result_series = pd.Series(0, index=fixtures_info_df['fixture_id'].unique(), name='abs_shots_on_target_diff_last_20_min')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(0).astype(int)

    # Define conditions for a shot on target based on original logic
    # This part is crucial and must match how SOTs are identified.
    sot_condition = (
        ((processed_events_df['event_type'] == 'Shot') & (processed_events_df['event_detail'] == 'Shot on Target')) |
        (processed_events_df['event_detail'] == 'Shot on target') | # common variation
        (processed_events_df['event_detail'] == 'On Target')      # common variation for event_type 'shot'
    )
    sot_events = processed_events_df[sot_condition].copy()
    
    if sot_events.empty:
        result_series = pd.Series(0, index=fixtures_info_df['fixture_id'].unique(), name='abs_shots_on_target_diff_last_20_min')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(0).astype(int)

    late_sot_events = sot_events[sot_events['time_elapsed'] >= 70].copy()

    if late_sot_events.empty:
        result_series = pd.Series(0, index=fixtures_info_df['fixture_id'].unique(), name='abs_shots_on_target_diff_last_20_min')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(0).astype(int)

    # Identify home and away SOTs
    late_sot_events['is_home_sot'] = late_sot_events['team_id'] == late_sot_events['home_team_id']
    late_sot_events['is_away_sot'] = late_sot_events['team_id'] == late_sot_events['away_team_id']

    sot_counts = late_sot_events.groupby('fixture_id').agg(
        home_sots_late=('is_home_sot', 'sum'),
        away_sots_late=('is_away_sot', 'sum')
    )

    abs_diff_sot_series = (sot_counts['home_sots_late'] - sot_counts['away_sots_late']).abs().astype(int)
    abs_diff_sot_series.name = 'abs_shots_on_target_diff_last_20_min'
    
    base_fixture_ids = fixtures_info_df.set_index('fixture_id').index
    result_series = abs_diff_sot_series.reindex(base_fixture_ids).fillna(0).astype(int)
    return result_series

def _calculate_total_yellow_cards_first_half_vectorized(processed_events_df: pd.DataFrame, fixtures_info_df: pd.DataFrame) -> pd.Series:
    """
    Calculates the total number of yellow cards in the first half (<= 45 min).
    Args:
        processed_events_df: DataFrame from _preprocess_events.
        fixtures_info_df: DataFrame with fixture_id for all relevant fixtures.
    Returns:
        pd.Series: Indexed by fixture_id, value is the count of yellow cards (int).
                    Defaults to 0 for fixtures with no first-half yellow cards.
    """
    if processed_events_df.empty or 'event_detail' not in processed_events_df.columns:
        result_series = pd.Series(0, index=fixtures_info_df['fixture_id'].unique(), name='total_yellow_cards_first_half')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(0).astype(int)

    # Filter for yellow card events in the first half
    first_half_yellow_cards = processed_events_df[
        (processed_events_df['event_detail'] == 'Yellow Card') &
        (processed_events_df['time_elapsed'] <= 45)
    ]

    if first_half_yellow_cards.empty:
        result_series = pd.Series(0, index=fixtures_info_df['fixture_id'].unique(), name='total_yellow_cards_first_half')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(0).astype(int)

    # Count yellow cards per fixture
    yc_counts_series = first_half_yellow_cards.groupby('fixture_id').size()
    yc_counts_series.name = 'total_yellow_cards_first_half'
    
    base_fixture_ids = fixtures_info_df.set_index('fixture_id').index
    result_series = yc_counts_series.reindex(base_fixture_ids).fillna(0).astype(int)
    return result_series

def _calculate_abs_goal_diff_0_30_min_vectorized(processed_events_df: pd.DataFrame, fixtures_info_df: pd.DataFrame) -> pd.Series:
    """
    Calculates absolute goal difference in the first 30 minutes (0-30 min).
    """
    scores_at_30_min = _get_score_at_time_vectorized(processed_events_df, 30, fixtures_info_df)
    abs_diff_series = (scores_at_30_min['home_goals_at_time'] - scores_at_30_min['away_goals_at_time']).abs().astype(int)
    abs_diff_series.name = 'abs_goal_diff_0_30_min'
    # _get_score_at_time_vectorized already ensures all fixtures from fixtures_info_df are present and defaults to 0 goals.
    return abs_diff_series

def _calculate_abs_goal_diff_30_60_min_vectorized(processed_events_df: pd.DataFrame, fixtures_info_df: pd.DataFrame) -> pd.Series:
    """
    Calculates absolute goal difference in the middle 30 minutes (30-60 min).
    """
    scores_at_30_min = _get_score_at_time_vectorized(processed_events_df, 30, fixtures_info_df)
    scores_at_60_min = _get_score_at_time_vectorized(processed_events_df, 60, fixtures_info_df)

    home_goals_in_period = scores_at_60_min['home_goals_at_time'] - scores_at_30_min['home_goals_at_time']
    away_goals_in_period = scores_at_60_min['away_goals_at_time'] - scores_at_30_min['away_goals_at_time']
    
    abs_diff_series = (home_goals_in_period - away_goals_in_period).abs().astype(int)
    abs_diff_series.name = 'abs_goal_diff_30_60_min'
    return abs_diff_series

def _calculate_abs_goal_diff_60_90_min_vectorized(processed_events_df: pd.DataFrame, fixtures_info_df: pd.DataFrame) -> pd.Series:
    """
    Calculates absolute goal difference in the last 30 regular minutes (60-90 min).
    Note: Original logic used max_time (max(90, fixture_events_df['time_elapsed'].max())).
            _get_score_at_time_vectorized uses a fixed time_limit. For 90min segment, using 90.
    """
    scores_at_60_min = _get_score_at_time_vectorized(processed_events_df, 60, fixtures_info_df)
    # For this segment, the original logic considered goals up to the actual max event time if > 90.
    # However, _get_score_at_time_vectorized takes a fixed limit. We use 90 for this segment end.
    # If strict adherence to original max_time is needed, _get_score_at_time_vectorized would need adjustment or a new variant.
    scores_at_90_min = _get_score_at_time_vectorized(processed_events_df, 90, fixtures_info_df)

    home_goals_in_period = scores_at_90_min['home_goals_at_time'] - scores_at_60_min['home_goals_at_time']
    away_goals_in_period = scores_at_90_min['away_goals_at_time'] - scores_at_60_min['away_goals_at_time']
    
    abs_diff_series = (home_goals_in_period - away_goals_in_period).abs().astype(int)
    abs_diff_series.name = 'abs_goal_diff_60_90_min'
    return abs_diff_series

def _calculate_abs_substitutions_diff_overall_vectorized(processed_events_df: pd.DataFrame, fixtures_info_df: pd.DataFrame) -> pd.Series:
    """
    Calculates the absolute difference in total substitutions made by home and away teams.
    Args:
        processed_events_df: DataFrame from _preprocess_events.
        fixtures_info_df: DataFrame with fixture_id, home_team_id, away_team_id.
    Returns:
        pd.Series: Indexed by fixture_id, value is the absolute difference (int).
                    Defaults to 0 for fixtures with no substitutions.
    """
    if processed_events_df.empty or 'subst' not in processed_events_df['event_type'].unique():
        result_series = pd.Series(0, index=fixtures_info_df['fixture_id'].unique(), name='abs_substitutions_diff_overall')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(0).astype(int)

    sub_events = processed_events_df[processed_events_df['event_type'] == 'subst'].copy()
    if sub_events.empty:
        result_series = pd.Series(0, index=fixtures_info_df['fixture_id'].unique(), name='abs_substitutions_diff_overall')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(0).astype(int)

    # Identify home and away substitutions
    # Ensure home_team_id and away_team_id are present from the merge in the main function
    if not all(col in sub_events.columns for col in ['home_team_id', 'away_team_id']):
         # This should not happen if main merge was successful
        print("Warning: home_team_id or away_team_id missing in sub_events. Cannot calculate abs_substitutions_diff_overall accurately. Returning 0s.")
        result_series = pd.Series(0, index=fixtures_info_df['fixture_id'].unique(), name='abs_substitutions_diff_overall')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(0).astype(int)

    sub_events['is_home_sub'] = sub_events['team_id'] == sub_events['home_team_id']
    sub_events['is_away_sub'] = sub_events['team_id'] == sub_events['away_team_id']

    sub_counts = sub_events.groupby('fixture_id').agg(
        home_subs=('is_home_sub', 'sum'),
        away_subs=('is_away_sub', 'sum')
    )

    abs_diff_subs_series = (sub_counts['home_subs'] - sub_counts['away_subs']).abs().astype(int)
    abs_diff_subs_series.name = 'abs_substitutions_diff_overall'
    
    base_fixture_ids = fixtures_info_df.set_index('fixture_id').index
    result_series = abs_diff_subs_series.reindex(base_fixture_ids).fillna(0).astype(int)
    return result_series

def _proportion_time_tied_fixture(fixture_events: pd.DataFrame) -> float:
    """
    Helper for a single fixture to calculate the proportion of game time the score was tied.
    Assumes fixture_events are sorted and contain home_team_id, away_team_id.
    """
    if fixture_events.empty:
        return 1.0 # No events, score 0-0 for the whole match (assumed 90 min)

    home_team_id = fixture_events['home_team_id'].iloc[0]
    away_team_id = fixture_events['away_team_id'].iloc[0]

    goal_event_times = fixture_events[fixture_events['is_goal']]['time_elapsed'].unique()
    
    match_end_time = 90.0
    if not fixture_events.empty():
        max_event_time = fixture_events['time_elapsed'].max()
        if max_event_time > match_end_time:
            match_end_time = max_event_time
            
    # Critical time points: start, goal times, and match end.
    time_points = sorted(list(set([0.0] + list(goal_event_times) + [match_end_time])))
    time_points = sorted(list(set(tp for tp in time_points if 0 <= tp <= match_end_time)))
    if not time_points or time_points[-1] < match_end_time:
        if not any(tp == match_end_time for tp in time_points):
            time_points.append(match_end_time)
        time_points = sorted(list(set(time_points)))
    if 0.0 not in time_points:
        time_points.insert(0,0.0) # Ensure 0 is the start

    if not time_points or match_end_time == 0: # No duration or invalid match end time
        # If match end time is 0, but events might exist, this is tricky.
        # Let's assume if match_end_time is 0, proportion is 1.0 if no goals, 0.0 if goals (implies some duration)
        # However, the current logic for match_end_time means it's at least 90 unless events are truly all at 0.
        # Sticking to original: if match_end_time is 0, what was score? Assume 0-0 -> 1.0
        # More robust: if match_end_time is effectively 0, and no goals, it implies 1.0. If goals, implies some duration, this case is odd.
        # For now, if match_end_time is 0, return 1.0 (as if 0-0 at t=0 for 0 duration)
        # This covers fixture_events being non-empty but all at time 0 with match_end_time also 0.
        # If fixture_events is empty, already returned 1.0.
        # If match_end_time is 0 based on max_event_time being 0, and events exist at t=0:
        # Calculate score at t=0. If tied, 1.0, else 0.0.
        if match_end_time == 0:
            h_goals_at_0, a_goals_at_0 = 0, 0
            if not fixture_events[fixture_events['time_elapsed'] == 0].empty:
                goals_at_0 = fixture_events[(fixture_events['time_elapsed'] == 0) & (fixture_events['is_goal'])].copy()
                if not goals_at_0.empty:
                    h_goals_at_0 = goals_at_0[goals_at_0['team_id'] == home_team_id].shape[0]
                    a_goals_at_0 = goals_at_0[goals_at_0['team_id'] == away_team_id].shape[0]
            return 1.0 if h_goals_at_0 == a_goals_at_0 else 0.0

    total_time_score_tied = 0.0
    # Pre-calculate cumulative scores for this fixture
    _fixture_events_sorted = fixture_events.sort_values(by='time_elapsed').copy()
    _fixture_events_sorted['is_home_g'] = (_fixture_events_sorted['is_goal']) & (_fixture_events_sorted['team_id'] == home_team_id)
    _fixture_events_sorted['is_away_g'] = (_fixture_events_sorted['is_goal']) & (_fixture_events_sorted['team_id'] == away_team_id)
    _fixture_events_sorted['cum_home_g'] = _fixture_events_sorted['is_home_g'].cumsum()
    _fixture_events_sorted['cum_away_g'] = _fixture_events_sorted['is_away_g'].cumsum()

    def get_score_at_t_for_proportion(t_limit, _events_with_cum_scores):
        relevant_events = _events_with_cum_scores[_events_with_cum_scores['time_elapsed'] <= t_limit]
        if relevant_events.empty:
            return 0, 0
        last_event_at_t = relevant_events.iloc[-1]
        return last_event_at_t['cum_home_g'], last_event_at_t['cum_away_g']

    # Iterate through segments defined by the unique time points
    for i in range(len(time_points) -1): # Iterate up to the second to last point
        segment_start_time = time_points[i]
        segment_end_time = time_points[i+1]
        segment_duration = segment_end_time - segment_start_time

        if segment_duration <= 0: # Should not happen if time_points are unique and sorted
            continue

        # Score at the START of this segment (which is the score at segment_start_time)
        h_goals_at_segment_start, a_goals_at_segment_start = get_score_at_t_for_proportion(segment_start_time, _fixture_events_sorted)

        if h_goals_at_segment_start == a_goals_at_segment_start:
            total_time_score_tied += segment_duration
            
    # Handle the very first state at t=0. If time_points starts with 0, the first segment is [0, time_points[1]].
    # The logic above correctly uses score at t=0 (0-0) for the first segment.

    return round(total_time_score_tied / match_end_time, 3) if match_end_time > 0 else (1.0 if total_time_score_tied == 0 else 0.0) # Avoid div by zero, though match_end_time logic tries to prevent it.
                                                                                                                                    # If match_end_time is 0 and total_time_score_tied is also 0 (e.g. only event at t=0 was a goal), this is 1.0. Correct as it was tied for its non-duration.

def _calculate_proportion_of_game_time_score_is_tied_at_end_vectorized(processed_events_df: pd.DataFrame, fixtures_info_df: pd.DataFrame) -> pd.Series:
    """
    Calculates the proportion of game time the score was tied.
    Args:
        processed_events_df: DataFrame from _preprocess_events.
        fixtures_info_df: DataFrame with fixture_id, home_team_id, away_team_id.
    Returns:
        pd.Series: Indexed by fixture_id, value is the proportion (float).
                    Defaults to 1.0 for fixtures with no events (tied 0-0 for full match).
    """
    if processed_events_df.empty:
        result_series = pd.Series(1.0, index=fixtures_info_df['fixture_id'].unique(), name='proportion_of_game_time_score_is_tied_at_end')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(1.0)

    if not all(col in processed_events_df.columns for col in ['home_team_id', 'away_team_id']):
        print("Warning: Essential columns missing for proportion_of_game_time_score_is_tied. Returning NaNs or 0s might be better here.")
        # Defaulting to NaN as calculation is unreliable
        result_series = pd.Series(np.nan, index=fixtures_info_df['fixture_id'].unique(), name='proportion_of_game_time_score_is_tied_at_end')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(np.nan)

    try:
        proportion_series = processed_events_df.groupby('fixture_id').apply(lambda x: _proportion_time_tied_fixture(x))
        proportion_series.name = 'proportion_of_game_time_score_is_tied_at_end'
        
        base_fixture_ids = fixtures_info_df.set_index('fixture_id').index
        result_series = proportion_series.reindex(base_fixture_ids).fillna(1.0) # Default for no events is 1.0 (tied 0-0)
        return result_series
    except Exception as e:
        print(f"Error calculating proportion_of_game_time_score_is_tied_at_end: {str(e)}. Filling with NaNs.")
        result_series = pd.Series(np.nan, index=fixtures_info_df['fixture_id'].unique(), name='proportion_of_game_time_score_is_tied_at_end')
        return result_series.reindex(fixtures_info_df.set_index('fixture_id').index).fillna(np.nan)

def _get_fixtures_from_db() -> pd.DataFrame:
    """Fetches fixtures data from the PostgreSQL database."""
    load_dotenv() # Load environment variables from .env file

    db_host = os.getenv("POSTGRES_HOST")
    db_name = os.getenv("POSTGRES_DB")
    db_user = os.getenv("POSTGRES_USER")
    db_password = os.getenv("POSTGRES_PASSWORD")
    db_port = os.getenv("POSTGRES_PORT", "5432")

    conn = None
    fixtures_df = pd.DataFrame()

    # User-provided SQL query
    sql_query = """
    SELECT 
        fixture_id, referee, timezone, "date", "timestamp", 
        period_first, period_second, venue_id, venue_name, venue_city, 
        status_long, status_short, status_elapsed, status_extra, league_id, 
        league_name, league_country, league_logo, league_flag, league_season, 
        league_round, league_standings, home_team_id, home_team_name, home_team_logo, 
        home_team_winner as home_win, away_team_id, away_team_name, away_team_logo, away_team_winner as away_win, 
        home_goals, away_goals, home_halftime_goals, away_halftime_goals, 
        home_extratime_goals, away_extratime_goals, home_penalty_goals, 
        away_penalty_goals, home_shots_on_goal, home_shots_off_goal, home_total_shots, home_blocked_shots, 
        home_shots_insidebox, home_shots_outsidebox, home_fouls, home_corner_kicks, home_offsides, 
        home_ball_possession/100 as home_ball_possession_accuracy,
        home_yellow_cards, home_red_cards, home_goalkeeper_saves, home_total_passes, home_passes_accurate, 
        away_shots_on_goal, away_shots_off_goal, away_total_shots, away_blocked_shots, away_shots_insidebox, 
        away_shots_outsidebox, away_fouls, away_corner_kicks, away_offsides, 
        away_ball_possession/100 as away_ball_possession_accuracy, away_yellow_cards, 
        away_red_cards, away_goalkeeper_saves, away_total_passes, away_passes_accurate, 
        home_passes_percent/100 as home_passes_accuracy, away_passes_percent/100 as away_passes_accuracy
    FROM api_football.fixtures
    WHERE 
        home_goals IS NOT NULL and home_total_shots is not null;
    """

    sample_fallback_fixtures_data = {
        'fixture_id': [1, 2, 3, 4],
        'home_team_id': [101, 103, 105, 107],
        'away_team_id': [102, 104, 106, 108]
    }

    try:
        if not all([db_host, db_name, db_user, db_password, db_port]):
            print("Database environment variables not fully set. Falling back to sample fixtures data.")
            return pd.DataFrame(sample_fallback_fixtures_data)

        conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=db_port
        )
        print("Successfully connected to PostgreSQL database for fixtures.")
        fixtures_df = pd.read_sql_query(sql_query, conn)
        print(f"Successfully fetched {len(fixtures_df)} fixtures from the database.")
        
        if fixtures_df.empty:
            print("Query returned no fixtures. Falling back to sample fixtures data.")
            return pd.DataFrame(sample_fallback_fixtures_data)
            
        # Ensure necessary columns for calculator are present
        required_cols = ['fixture_id', 'home_team_id', 'away_team_id']
        if not all(col in fixtures_df.columns for col in required_cols):
            print(f"Fetched fixtures_df is missing one of required columns: {required_cols}. Falling back to sample data.")
            return pd.DataFrame(sample_fallback_fixtures_data)

    except psycopg2.Error as e:
        print(f"Error connecting to or querying PostgreSQL database: {e}")
        print("Falling back to sample fixtures_df due to DB error.")
        return pd.DataFrame(sample_fallback_fixtures_data)
    except Exception as e:
        print(f"An unexpected error occurred during DB operation: {e}")
        print("Falling back to sample fixtures_df due to unexpected error.")
        return pd.DataFrame(sample_fallback_fixtures_data)
    finally:
        if conn:
            conn.close()
            print("PostgreSQL connection closed.")
    
    return fixtures_df

def _get_events_from_db() -> pd.DataFrame:
    """Fetches events data from the PostgreSQL database."""
    load_dotenv() # Ensure env vars are loaded

    db_host = os.getenv("POSTGRES_HOST")
    db_name = os.getenv("POSTGRES_DB")
    db_user = os.getenv("POSTGRES_USER")
    db_password = os.getenv("POSTGRES_PASSWORD")
    db_port = os.getenv("POSTGRES_PORT", "5432")

    conn = None
    events_df = pd.DataFrame()

    sql_query = """
    SELECT 
        e.*,
        f.home_team_id,
        f.away_team_id
    FROM api_football.events e
    LEFT JOIN api_football.fixtures f ON e.fixture_id = f.fixture_id;
    """

    sample_fallback_events_data = {
        'fixture_id':   [1,   1,   1,    1,   1,   2,   2,    2,    3],
        'time_elapsed': [30,  40,  70,   78,  85,  25,  60,   65,   10],
        'event_type':   ['Card','Goal','subst','Goal','subst','Shot','Goal','Card','subst'],
        'event_detail': ['Yellow Card', None, None, None, None, 'On Target',None,'Red Card', None],
        'team_id':      [101, 101, 101,  102, 102, 103, 103,  104,  105],
        # Sample data needs home_team_id and away_team_id if DB fails
        'home_team_id': [101, 101, 101,  101, 101, 103, 103,  103,  105],
        'away_team_id': [102, 102, 102,  102, 102, 104, 104,  104,  106]
    }

    try:
        if not all([db_host, db_name, db_user, db_password, db_port]):
            print("Database environment variables not fully set. Falling back to sample events data.")
            return pd.DataFrame(sample_fallback_events_data)

        conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=db_port
        )
        print("Successfully connected to PostgreSQL database for events.")
        events_df = pd.read_sql_query(sql_query, conn)
        # Add time_extra to time_elapsed if time_extra exists and is not null
        if 'time_extra' in events_df.columns:
            events_df['time_elapsed'] = events_df.apply(
                lambda row: row['time_elapsed'] + row['time_extra'] 
                if pd.notnull(row['time_extra']) else row['time_elapsed'], 
                axis=1
            )
            events_df = events_df.drop(columns=['time_extra'])
        print(f"Successfully fetched {len(events_df)} events from the database.")

        if events_df.empty:
            print("Query returned no events. Falling back to sample events data.")
            return pd.DataFrame(sample_fallback_events_data)
        
        required_cols = ['fixture_id', 'time_elapsed', 'event_type', 'event_detail', 'team_id', 'home_team_id', 'away_team_id']
        if not all(col in events_df.columns for col in required_cols):
            print(f"Fetched events_df is missing one of required columns: {required_cols}. Falling back to sample data.")
            return pd.DataFrame(sample_fallback_events_data)

    except psycopg2.Error as e:
        print(f"Error connecting to or querying PostgreSQL database for events: {e}")
        print("Falling back to sample events_df due to DB error.")
        return pd.DataFrame(sample_fallback_events_data)
    except Exception as e:
        print(f"An unexpected error occurred during DB operation for events: {e}")
        print("Falling back to sample events_df due to unexpected error.")
        return pd.DataFrame(sample_fallback_events_data)
    finally:
        if conn:
            conn.close()
            print("PostgreSQL connection for events closed.")
            
    return events_df

def main_test():
    """Main function for testing the event feature calculation."""
    
    # Get fixtures from DB or fallback to sample
    fixtures_df = _get_fixtures_from_db()

    # Get events from DB or fallback to sample
    events_df = _get_events_from_db()

    print("--- Fixtures Data (from DB or sample) ---")
    print(fixtures_df.head())
    print(f"Total fixtures loaded: {len(fixtures_df)}")
    print("\n--- Events Data (from DB or sample) ---")
    print(events_df.head())
    print(f"Total events loaded: {len(events_df)}")

    # Calculate features
    if not fixtures_df.empty:
        event_features = calculate_event_features_vectorized(events_df.copy(), fixtures_df.copy()) # Pass copies

        print("\n--- Calculated Event Features ---")
        export_to_xlsx_fast(event_features, "data/Create_data/data_files/base/event_features.xlsx")
    else:
        print("\n--- Could not load fixtures_df, skipping feature calculation. ---")

if __name__ == "__main__":
    main_test() 