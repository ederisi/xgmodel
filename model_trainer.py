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
        self.categories = {}
        self.goalie_options = []
        
        # Includes features: location, handedness, off-wing, team and goalie tier
        self.features = [
            'arenaAdjustedShotDistance', 
            'arenaAdjustedXCordABS', 
            'arenaAdjustedYCordAbs',
            'shooterLeftRight',
            'offWing',
            'teamCode',
            'goalieTier'
        ]
        self.goal_col = 'goal'

    def load_data(self):
        """Loads data from shots.csv and classifies goalies (Top 5, Bottom 5, Average Tendie=.)"""
        if self.df is None:
            print(f"Loading data from '{self.data_path}'...")
            self.df = pd.read_csv(self.data_path)

            # Check for goal column variations
            if 'isGoal' in self.df.columns and 'goal' not in self.df.columns:
                self.goal_col = 'isGoal'
            # Prepares for goalie tier classification (Top 5, Bottom 5, Average Tendie)
            self._prepare_goalie_tiers()
        return self.df

    def _prepare_goalie_tiers(self):
        """Separates goalies into Top 5, Bottom 5 and Average Tendie.
        Shot amounts are also taken to account."""

        if 'goalieNameForShot' not in self.df.columns:
            self.df['goalieTier'] = "Average Tendie"
            self.goalie_options = ["Average Tendie"]
            return

        # Count stats for goalies (atleast 200 shots to qualify).
        g_stats = self.df.groupby('goalieNameForShot').agg(
            shots=(self.goal_col, 'count'),
            goals=(self.goal_col, 'sum')
        )
        qualified = g_stats[g_stats['shots'] >= 200].copy()
        qualified['goal_rate'] = qualified['goals'] / qualified['shots']

        # Least amount of goals per shot = best goalies
        top_5 = qualified.sort_values(by='goal_rate', ascending=True).head(5).index.tolist()
        # Largest amount of goals per shot = worst goalies
        bottom_5 = qualified.sort_values(by='goal_rate', ascending=False).head(5).index.tolist()

        print(f"\n[+] Top 5 Goalies (Elite): {top_5}")
        print(f"[-] Bottom 5 Goalies (Vulnerable): {bottom_5}")

        special_goalies = set(top_5 + bottom_5)
        self.df['goalieTier'] = self.df['goalieNameForShot'].apply(
            lambda x: x if x in special_goalies else "Average Tendie"
        )
        
        self.goalie_options = ["Average Tendie"] + sorted(list(special_goalies))

    def train_model(self):
        """Trains the XGBoost model with team and goalie weightings."""
        df = self.load_data()
        
        cat_cols = ['shooterLeftRight', 'teamCode', 'goalieTier']
        active_features = [f for f in self.features if f in df.columns]

        # Drop rows with missing feature values or goal column
        clean_df = df.dropna(subset=active_features + [self.goal_col]).copy()

        # Transform categorical columns to category dtype for XGBoost.
        for col in cat_cols:
            if col in clean_df.columns:
                clean_df[col] = clean_df[col].astype('category')
                self.categories[col] = clean_df[col].cat.categories

        X = clean_df[active_features]
        y = clean_df[self.goal_col]

        # Store active features for later use in prediction
        self.active_features = active_features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"\nTraining XGBoost xG model using features: {active_features}...")
        self.model = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=4, 
            random_state=42,
            enable_categorical=True,
            eval_metric='logloss'
        )
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict_proba(X_test)[:, 1]
        loss = log_loss(y_test, preds)
        brier = brier_score_loss(y_test, preds)
        
        print(f"Model Trained Successfully!")
        print(f" -> Log Loss: {loss:.4f} | Brier Score: {brier:.4f}")
        return self.model

    def predict_shot(self, distance, x_coord, y_coord, handedness, off_wing, team_code, goalie_tier):
        """Predicts the expected goals (xG) for a single shot based on the provided parameters."""
        if self.model is None:
            self.train_model()
            
        input_data = pd.DataFrame({
            'arenaAdjustedShotDistance': [distance],
            'arenaAdjustedXCordABS': [x_coord],
            'arenaAdjustedYCordAbs': [abs(y_coord)],
            'shooterLeftRight': pd.Categorical([handedness], categories=self.categories.get('shooterLeftRight')),
            'offWing': [off_wing],
            'teamCode': pd.Categorical([team_code], categories=self.categories.get('teamCode')),
            'goalieTier': pd.Categorical([goalie_tier], categories=self.categories.get('goalieTier'))
        })
        
        input_data = input_data[[f for f in self.active_features if f in input_data.columns]]
        return float(self.model.predict_proba(input_data)[:, 1][0])

    def get_team_leaderboard(self):
        """Calculates team-specific xG statistics and returns a leaderboard DataFrame."""
        if self.model is None:
            self.train_model()
            
        clean_df = self.df.dropna(subset=self.active_features + [self.goal_col]).copy()
        for col, cats in self.categories.items():
            if col in clean_df.columns:
                clean_df[col] = pd.Categorical(clean_df[col], categories=cats)
                
        X = clean_df[self.active_features]
        clean_df['xG'] = self.model.predict_proba(X)[:, 1]
        
        team_col = 'teamCode' if 'teamCode' in clean_df.columns else 'shooterTeamCode'
        
        team_stats = clean_df.groupby(team_col, observed=False).agg(
            Total_Shots=(self.goal_col, 'count'),
            Actual_Goals=(self.goal_col, 'sum'),
            Total_xG=('xG', 'sum')
        ).reset_index()
        
        team_stats['Goals_Above_Expected'] = (team_stats['Actual_Goals'] - team_stats['Total_xG']).round(2)
        team_stats['Total_xG'] = team_stats['Total_xG'].round(2)
        
        return team_stats.sort_values(by='Total_xG', ascending=False)