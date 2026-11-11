import pandas as pd
import numpy as np
from xgboost import xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss

class XGModelTrainer:
    def __init__(self, csv_path="shots.csv"):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.goal_col = 'isGoal' if ('isGoal' in self.df.columns and 'goal' not in self.df.columns) else 'goal'
        
        # Reorder the DataFrame by game_id and time if those columns exist (for later use in LSTM and GRU models)
        if 'game_id' in self.df.columns and 'time' in self.df.columns:
            self.df = self.df.sort_values(['game_id', 'time']).reset_index(drop=True)

        self._prepare_goalie_tiers()

        # all features to be used in the model, including categorical features
        self.features = [
            'arenaAdjustedShotDistance',
            'arenaAdjustedXCordABS',
            'arenaAdjustedYCordAbs',
            'shotAngleAdjusted',
            'shotType',
            'shotRebound',
            'shotRush',
            'shooterLeftRight',
            'offWing',
            'teamCode',
            'goalieTier'
        ]
        self.active_features = [f for f in self.features if f in self.df.columns]
        self.cat_cols = ['shooterLeftRight', 'teamCode', 'goalieTier', 'shotType']
        self.categories = {}
        self.model = None

    def _prepare_goalie_tiers(self):
        if 'goalieNameForShot' not in self.df.columns:
            self.df['goalieTier'] = "Average Tendie"
            self.goalie_options = ["Average Tendie"]
            return

        counts = self.df['goalieNameForShot'].value_counts()
        valid_goalies = counts[counts >= 200].index
        sub = self.df[self.df['goalieNameForShot'].isin(valid_goalies)]
        
        rates = sub.groupby('goalieNameForShot')[self.goal_col].mean().sort_values()
        top5 = list(rates.head(5).index)
        bot5 = list(rates.tail(5).index)
        special = set(top5 + bot5)

        self.df['goalieTier'] = self.df['goalieNameForShot'].apply(
            lambda x: x if x in special else "Average Tendie"
        )
        self.goalie_options = ["Average Tendie"] + top5 + bot5

    def train_model(self):
        clean_df = self.df.dropna(subset=self.active_features + [self.goal_col]).copy()
        
        for col in self.cat_cols:
            if col in clean_df.columns:
                clean_df[col] = clean_df[col].astype('category')
                self.categories[col] = list(clean_df[col].cat.categories)

        X = clean_df[self.active_features]
        y = clean_df[self.goal_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            enable_categorical=True,
            eval_metric='logloss',
            random_state=42
        )
        self.model.fit(X_train, y_train)

        preds = self.model.predict_proba(X_test)[:, 1]
        print(f" -> XGBoost Log Loss: {log_loss(y_test, preds):.4f} | Brier Score: {brier_score_loss(y_test, preds):.4f}")
        return self.model

    def predict_shot(self, distance, x_coord, y_coord, handedness, off_wing, team_code, goalie_tier,
                     shot_type="WRIST", is_rebound=0, is_rush=0):
        if self.model is None:
            self.train_model()

        # Calculate shot angle based on x and y coordinates
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
        
        input_df = pd.DataFrame([{k: row_dict[k] for k in self.active_features}])
        for col, cats in self.categories.items():
            if col in input_df.columns:
                input_df[col] = pd.Categorical(input_df[col], categories=cats)

        return float(self.model.predict_proba(input_df)[0, 1])