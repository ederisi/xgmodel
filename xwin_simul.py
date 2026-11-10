# Calculates the win probabilities for a given NHL game based on the xG values of each shot in the game.
# This is done in the monte carlo simulation style, where each shot is treated as a Bernoulli trial with the probability of scoring equal to the xG value of that shot.
import numpy as np
import pandas as pd

class GameWinSimulator:
    def __init__(self, xgb_trainer):
        self.trainer = xgb_trainer
        self.df = xgb_trainer.df

    def get_available_games(self, limit=150):
        """Searches the dataframe for unique game IDs and 
        returns a list of tuples containing the game ID and a string representation of the matchup."""
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
        """Computes "vectorized" Monte Carlo -simulation for shot's xG-lists."""
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
        """Simulates the win probabilities for a specific 
        game based on the xG values of each shot in that game."""
        game_df = self.df[self.df['game_id'] == game_id].dropna(
            subset=self.trainer.active_features + [self.trainer.goal_col]
        ).copy()
        
        for col, cats in self.trainer.categories.items():
            if col in game_df.columns:
                game_df[col] = pd.Categorical(game_df[col], categories=cats)

        X = game_df[self.trainer.active_features]
        game_df['xG'] = self.trainer.model.predict_proba(X)[:, 1]

        home_team = game_df['homeTeamCode'].iloc[0] if 'homeTeamCode' in game_df.columns else "Home"
        away_team = game_df['awayTeamCode'].iloc[0] if 'awayTeamCode' in game_df.columns else "Away"

        if 'isHomeTeam' in game_df.columns:
            home_mask = game_df['isHomeTeam'] == 1
        else:
            team_col = 'teamCode' if 'teamCode' in game_df.columns else 'shooterTeamCode'
            home_mask = game_df[team_col] == home_team

        home_xg = game_df.loc[home_mask, 'xG'].values
        away_xg = game_df.loc[~home_mask, 'xG'].values

        actual_home_goals = int(game_df.loc[home_mask, self.trainer.goal_col].sum())
        actual_away_goals = int(game_df.loc[~home_mask, self.trainer.goal_col].sum())

        sim_res = self.simulate_from_xg_arrays(home_xg, away_xg, num_simulations)
        sim_res.update({
            "home_team": home_team,
            "away_team": away_team,
            "actual_home_goals": actual_home_goals,
            "actual_away_goals": actual_away_goals,
            "home_shots": len(home_xg),
            "away_shots": len(away_xg)
        })
        return sim_res