import numpy as np


def calculate_draw_k_factor(league_data, logger):
    """
    Calculate Draw-specific K-factor for a league based on draw predictability.
    Higher K-factor = more predictable draw patterns
    Lower K-factor = more volatile/unpredictable draws
    """
    try:
        # 1. Draw Rate Stability (most important for draws)
        draw_rates = []
        for team_id in league_data['home_encoded'].unique():
            home_draws = league_data[league_data['home_encoded'] == team_id]['Home_draws'].iloc[0]
            home_matches = league_data[league_data['home_encoded'] == team_id]['Home_team_matches'].iloc[0]
            away_draws = league_data[league_data['away_encoded'] == team_id]['Away_draws'].iloc[0] 
            away_matches = league_data[league_data['away_encoded'] == team_id]['Away_team_matches'].iloc[0]
            
            if home_matches > 0 and away_matches > 0:
                team_draw_rate = (home_draws + away_draws) / (home_matches + away_matches)
                draw_rates.append(team_draw_rate)
        
        draw_rate_std = np.std(draw_rates) if draw_rates else 0.5
        draw_stability_factor = 1 - (draw_rate_std / 0.5)  # 0.5 = max possible std
        
        # 2. Goal Difference Concentration (draws happen with small goal differences)
        goal_diff_data = []
        for _, row in league_data.iterrows():
            if row['match_outcome'] == 2:  # Draw matches only
                home_goals = row.get('home_goals', 0)
                away_goals = row.get('away_goals', 0)
                goal_diff_data.append(abs(home_goals - away_goals))
        
        if goal_diff_data:
            # More 0-0, 1-1, 2-2 draws = higher predictability
            zero_diff_rate = sum(1 for diff in goal_diff_data if diff == 0) / len(goal_diff_data)
            small_diff_rate = sum(1 for diff in goal_diff_data if diff <= 1) / len(goal_diff_data)
            goal_pattern_factor = (zero_diff_rate * 0.6) + (small_diff_rate * 0.4)
        else:
            goal_pattern_factor = 0.5
        
        # 3. Temporal Draw Distribution (consistent vs. clustered draws)
        league_draws = league_data[league_data['match_outcome'] == 2]
        if len(league_draws) > 5:
            # Calculate draw frequency consistency across season
            league_draws_sorted = league_draws.sort_values('Date')
            draw_intervals = []
            for i in range(1, len(league_draws_sorted)):
                interval = (league_draws_sorted.iloc[i]['Date'] - 
                            league_draws_sorted.iloc[i-1]['Date']).days
                draw_intervals.append(interval)
            
            if draw_intervals:
                interval_std = np.std(draw_intervals)
                # Lower std = more consistent draw timing = higher predictability
                temporal_factor = max(0, 1 - (interval_std / 30))  # 30 days normalization
            else:
                temporal_factor = 0.5
        else:
            temporal_factor = 0.5
        
        # 4. ELO Difference in Draw Matches (balanced teams = more draws)
        draw_elo_diffs = []
        for _, row in league_data.iterrows():
            if row['match_outcome'] == 2 and 'home_team_elo' in row and 'away_team_elo' in row:
                elo_diff = abs(row['home_team_elo'] - row['away_team_elo'])
                draw_elo_diffs.append(elo_diff)
        
        if draw_elo_diffs:
            avg_draw_elo_diff = np.mean(draw_elo_diffs)
            # Smaller ELO differences in draws = more predictable
            elo_balance_factor = max(0, 1 - (avg_draw_elo_diff / 200))  # 200 ELO normalization
        else:
            elo_balance_factor = 0.5
        
        # Combine factors with weights optimized for draw prediction
        draw_k_factor = 15 + (25 * (
            draw_stability_factor * 0.4 +      # Most important
            goal_pattern_factor * 0.3 +        # Second most important  
            temporal_factor * 0.2 +            # Timing consistency
            elo_balance_factor * 0.1           # Team balance
        ))
        
        return round(draw_k_factor, 2)
        
    except Exception as e:
        logger.error(f"Error calculating draw K-factor: {str(e)}")
        return 25  # Default draw K-factor