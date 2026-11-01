import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss

class XGModelTrainer:
    def __init__(self, data_path="shots.csv"):
        self.data_path = data_path
        self.model = None
        self.df = None
        
        # Features including physical location, handedness, off-wing context, and goalie data!
        self.features = [
            'arenaAdjustedShotDistance', 
            'arenaAdjustedXCordABS', 
            'arenaAdjustedYCordAbs',
            'shooterLeftRight',
            'offWing',
            'goalieNameForShot'
        ]
        self.goal_col = 'goal'

    def load_data(self):
        """Loads and cleans the dataset."""
        if self.df is None:
            print(f"Loading data from '{self.data_path}'...")
            self.df = pd.read_csv(self.data_path)
            if 'isGoal' in self.df.columns and 'goal' not in self.df.columns:
                self.goal_col = 'isGoal'
        return self.df

    def train_model(self):
        """Trains the XGBoost classifier using real categorical data for goalies and handedness."""
        df = self.load_data()
        
        # Ensure categorical features are formatted properly for XGBoost
        for col in ['shooterLeftRight', 'goalieNameForShot']:
            if col in df.columns:
                df[col] = df[col].astype('category')

        # Filter features that actually exist in the CSV
        active_features = [f for f in self.features if f in df.columns]
        
        clean_df = df.dropna(subset=active_features + [self.goal_col]).copy()
        X = clean_df[active_features]
        y = clean_df[self.goal_col]

        self.active_features = active_features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Training XGBoost xG model using features: {active_features}...")
        self.model = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=4, 
            random_state=42,
            enable_categorical=True,
            eval_metric='logloss'
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate model performance
        preds = self.model.predict_proba(X_test)[:, 1]
        loss = log_loss(y_test, preds)
        brier = brier_score_loss(y_test, preds)
        
        print(f"Model Trained Successfully!")
        print(f" -> Log Loss: {loss:.4f} | Brier Score: {brier:.4f}")
        return self.model

    def predict_shot(self, distance, x_coord, y_coord, handedness, off_wing, goalie_name):
        """Predicts the xG probability for a shot using specific parameters."""
        if self.model is None:
            self.train_model()
            
        input_data = pd.DataFrame({
            'arenaAdjustedShotDistance': [distance],
            'arenaAdjustedXCordABS': [x_coord],
            'arenaAdjustedYCordAbs': [abs(y_coord)],
            'shooterLeftRight': pd.Categorical([handedness]),
            'offWing': [off_wing],
            'goalieNameForShot': pd.Categorical([goalie_name])
        })
        
        # Keep only the features the model was trained on
        input_data = input_data[[f for f in self.active_features if f in input_data.columns]]
        return self.model.predict_proba(input_data)[:, 1][0]

    def get_team_leaderboard(self):
        """Calculates team-level xG using the fully trained model."""
        if self.model is None:
            self.train_model()
            
        clean_df = self.df.dropna(subset=self.active_features + [self.goal_col]).copy()
        X = clean_df[self.active_features]
        clean_df['xG'] = self.model.predict_proba(X)[:, 1]
        
        team_col = 'teamCode' if 'teamCode' in clean_df.columns else 'shooterTeamCode'
        
        team_stats = clean_df.groupby(team_col).agg(
            Total_Shots=(self.goal_col, 'count'),
            Actual_Goals=(self.goal_col, 'sum'),
            Total_xG=('xG', 'sum')
        ).reset_index()
        
        team_stats['Goals_Above_Expected'] = (team_stats['Actual_Goals'] - team_stats['Total_xG']).round(2)
        team_stats['Total_xG'] = team_stats['Total_xG'].round(2)
        
        return team_stats.sort_values(by='Total_xG', ascending=False)