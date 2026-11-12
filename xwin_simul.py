"""
Monte Carlo simulation for win probabilities based on xG values of shots in a game. This module provides the GameWinSimulator class, 
which can be used to simulate the outcome of a hockey game given the expected goals (xG) values for each shot taken by both teams. 
The simulation uses a vectorized approach to efficiently compute win probabilities over a large number of simulations.
"""
import numpy as np
import pandas as pd

class GameWinSimulator:
    """
    Class for simulating win probabilities in a hockey game based on expected goals (xG) values of shots.
    """
    def __init__(self, xgb_trainer):
        """
        Initialize the GameWinSimulator with a trained XGBoost model trainer.
        """
        self.trainer = xgb_trainer
        self.df = xgb_trainer.df

    def get_available_games(self, limit=150):
        """Searches the dataframe for unique game IDs and 
        returns a list of tuples containing the game ID and a string representation of the matchup.
        """
        if 'game_id' not in self.df.columns:
            return []
        
        cols = ['game_id']
        for c in ['homeTeamCode', 'awayTeamCode']:
            if c in self.df.columns:
                cols.append(c)
                
        games = self.df[cols].drop_duplicates().head(limit)
        game_list = []
        for _, row in games.iterrows():
            gid = row['game_id']
            home = row.get('homeTeamCode', 'HOME')
            away = row.get('awayTeamCode', 'AWAY')
            game_list.append((gid, f"Game {gid}: {away} @ {home}"))
        return game_list

    def simulate_from_xg_arrays(self, home_xg_arr, away_xg_arr, num_simulations=10000):
        """
        Computes "vectorized" Monte Carlo -simulation for shot's xG-lists.
        Every shot is treated as a Bernoulli trial with probability equal to its xG value.
        """
        # Simulate home and away goals based on xG values
        if len(home_xg_arr) > 0:
            home_matrix = np.random.rand(num_simulations, len(home_xg_arr)) < home_xg_arr
            home_goals = home_matrix.sum(axis=1)
        else:
            home_goals = np.zeros(num_simulations, dtype=int)

        if len(away_xg_arr) > 0:
            away_matrix = np.random.rand(num_simulations, len(away_xg_arr)) < away_xg_arr
            away_goals = away_matrix.sum(axis=1)
        else:
            away_goals = np.zeros(num_simulations, dtype=int)

        # Calculate win probabilities based on simulated goals
        home_reg_wins = np.sum(home_goals > away_goals)
        away_reg_wins = np.sum(away_goals > home_goals)
        ties = np.sum(home_goals == away_goals)

        # Overtime/shootout probabilities are not explicitly modeled here, 
        # but can be inferred from the number of ties.
        home_total_win_prob = (home_reg_wins + 0.5 * ties) / num_simulations * 100
        away_total_win_prob = (away_reg_wins + 0.5 * ties) / num_simulations * 100
        ot_prob = (ties / num_simulations) * 100

        return {
            "home_xG": float(np.sum(home_xg_arr)),
            "away_xG": float(np.sum(away_xg_arr)),
            "home_win_prob": float(home_total_win_prob),
            "away_win_prob": float(away_total_win_prob),
            "ot_prob": float(ot_prob),
            "home_goals_dist": home_goals,
            "away_goals_dist": away_goals
        }

    def simulate_real_game(self, game_id, num_simulations=10000):
        """
        Simulates the win probabilities for a specific 
        game based on the xG values of each shot in that game.
        """
        # Filter the dataframe for the specified game ID and drop rows with missing values in relevant columns
        game_df = self.df[self.df['game_id'] == game_id].dropna(
            subset=self.trainer.active_features + [self.trainer.goal_col]
        ).copy()

        # Set categorical columns to the same categories as used in the model training
        for col, cats in self.trainer.categories.items():
            if col in game_df.columns:
                game_df[col] = pd.Categorical(game_df[col], categories=cats)

        # Compute xG values for each shot in the game using the trained model
        X = game_df[self.trainer.active_features]
        game_df['xG'] = self.trainer.model.predict_proba(X)[:, 1]

        # Determine home and away teams and separate xG values accordingly
        home_team = game_df['homeTeamCode'].iloc[0] if 'homeTeamCode' in game_df.columns else "Home"
        away_team = game_df['awayTeamCode'].iloc[0] if 'awayTeamCode' in game_df.columns else "Away"

        if 'isHomeTeam' in game_df.columns:
            home_mask = game_df['isHomeTeam'] == 1
        else:
            team_col = 'teamCode' if 'teamCode' in game_df.columns else 'shooterTeamCode'
            home_mask = game_df[team_col] == home_team

        # Extract xG values for home and away teams
        home_xg = game_df.loc[home_mask, 'xG'].values
        away_xg = game_df.loc[~home_mask, 'xG'].values

        actual_home_goals = int(game_df.loc[home_mask, self.trainer.goal_col].sum())
        actual_away_goals = int(game_df.loc[~home_mask, self.trainer.goal_col].sum())

        # Run the Monte Carlo simulation based on the xG values of shots for both teams
        sim_res = self.simulate_from_xg_arrays(home_xg, away_xg, num_simulations)
        
        # Update the simulation results with additional game information
        sim_res.update({
            "home_team": home_team,
            "away_team": away_team,
            "actual_home_goals": actual_home_goals,
            "actual_away_goals": actual_away_goals,
            "home_shots": len(home_xg),
            "away_shots": len(away_xg)
        })
        return sim_res