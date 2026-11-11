import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SklearnPreprocessor:
    def __init__(self, df, active_features, goal_col, seq_len=4):
        self.active_features = active_features
        self.goal_col = goal_col
        self.seq_len = seq_len
        self.cat_cols = ['shooterLeftRight', 'teamCode', 'goalieTier', 'shotType']
        self.num_cols = [c for c in active_features if c not in self.cat_cols]
        
        clean_df = df.dropna(subset=active_features + [goal_col]).copy()
        if 'game_id' in clean_df.columns and 'time' in clean_df.columns:
            clean_df = clean_df.sort_values(['game_id', 'time']).reset_index(drop=True)

        cat_active = [c for c in self.cat_cols if c in active_features]
        num_active = [c for c in self.num_cols if c in active_features]

        df_encoded = pd.get_dummies(clean_df[active_features], columns=cat_active, dtype=float)
        self.encoded_columns = df_encoded.columns.tolist()

        self.scaler = StandardScaler()
        df_encoded[num_active] = self.scaler.fit_transform(df_encoded[num_active])

        X_all = df_encoded.values.astype(np.float32)
        y_all = clean_df[goal_col].values.astype(np.float32)
        game_ids = clean_df['game_id'].values if 'game_id' in clean_df.columns else np.zeros(len(clean_df))

        # Create also the 3D sequences for LSTM/GRU models
        X_seq_all = self.build_sequences(X_all, game_ids, self.seq_len)

        # shuffle=False to maintain the temporal order for LSTM/GRU models
        split_idx = int(len(X_all) * 0.8)
        self.X_train, self.X_test = X_all[:split_idx], X_all[split_idx:]
        self.y_train, self.y_test = y_all[:split_idx], y_all[split_idx:]
        self.X_seq_train, self.X_seq_test = X_seq_all[:split_idx], X_seq_all[split_idx:]

    @staticmethod
    def build_sequences(X_2d, game_ids, seq_len=4):
        """Builds a 3D tensor (N, seq_len, F) containing the previous shots from the same game."""
        N, F = X_2d.shape
        X_seq = np.zeros((N, seq_len, F), dtype=np.float32)
        for step in range(seq_len):
            shift = seq_len - 1 - step
            if shift == 0:
                X_seq[:, step, :] = X_2d
            else:
                shifted_X = np.roll(X_2d, shift, axis=0)
                shifted_games = np.roll(game_ids, shift, axis=0)
                same_game = (game_ids == shifted_games)
                same_game[:shift] = False
                X_seq[same_game, step, :] = shifted_X[same_game]
        return X_seq

    def transform_single(self, distance, x_coord, y_coord, handedness, off_wing, team_code, goalie_tier,
                         shot_type="WRIST", is_rebound=0, is_rush=0):
        dx = max(89.0 - float(x_coord), 1.0)
        angle = float(np.degrees(np.arctan2(abs(float(y_coord)), dx)))

        row_dict = {
            'arenaAdjustedShotDistance': float(distance),
            'arenaAdjustedXCordABS': float(x_coord),
            'arenaAdjustedYCordAbs': abs(float(y_coord)),
            'shotAngleAdjusted': angle,
            'shotType': shot_type,
            'shotRebound': int(is_rebound),
            'shotRush': int(is_rush),
            'shooterLeftRight': handedness,
            'offWing': int(off_wing),
            'teamCode': team_code,
            'goalieTier': goalie_tier
        }
        df_single = pd.DataFrame([{k: row_dict[k] for k in self.active_features}])
        return self.transform_batch(df_single)

    def transform_batch(self, df_new):
        cat_active = [c for c in self.cat_cols if c in self.active_features]
        num_active = [c for c in self.num_cols if c in self.active_features]

        df_encoded = pd.get_dummies(df_new[self.active_features], columns=cat_active, dtype=float)
        df_encoded = df_encoded.reindex(columns=self.encoded_columns, fill_value=0.0)
        df_encoded[num_active] = self.scaler.transform(df_encoded[num_active])

        return df_encoded.values.astype(np.float32)

    def transform_batch_sequences(self, df_new):
        """Transforms the new DataFrame into 3D sequences (N, seq_len, F) for LSTM/GRU models."""
        X_2d = self.transform_batch(df_new)
        game_ids = df_new['game_id'].values if 'game_id' in df_new.columns else np.zeros(len(df_new))
        return self.build_sequences(X_2d, game_ids, self.seq_len)