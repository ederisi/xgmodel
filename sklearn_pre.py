import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SklearnPreprocessor:
    def __init__(self, df, active_features, goal_col='goal'):
        self.df = df
        self.active_features = active_features
        self.goal_col = goal_col
        self.scaler = StandardScaler()
        self.encoded_columns = None
        self.num_cols = ['arenaAdjustedShotDistance', 'arenaAdjustedXCordABS', 'arenaAdjustedYCordAbs', 'offWing']
        self.cat_cols = ['shooterLeftRight', 'teamCode', 'goalieTier']
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self._prepare_data()

    def _prepare_data(self):
        """Transforms categories, scales numeric values and splits the data into training and test sets."""
        clean_df = self.df.dropna(subset=self.active_features + [self.goal_col]).copy()
        cat_active = [c for c in self.cat_cols if c in self.active_features]
        num_active = [c for c in self.num_cols if c in self.active_features]

        df_encoded = pd.get_dummies(clean_df[self.active_features], columns=cat_active, dtype=float)
        self.encoded_columns = df_encoded.columns
        df_encoded[num_active] = self.scaler.fit_transform(df_encoded[num_active])

        X = df_encoded.values.astype(np.float32)
        y = clean_df[self.goal_col].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def transform_single(self, distance, x_coord, y_coord, handedness, off_wing, team_code, goalie_tier):
        """Transforms the parameters of a single shot into the format required by the models."""
        input_df = pd.DataFrame({
            'arenaAdjustedShotDistance': [distance],
            'arenaAdjustedXCordABS': [x_coord],
            'arenaAdjustedYCordAbs': [abs(y_coord)],
            'shooterLeftRight': [handedness],
            'offWing': [off_wing],
            'teamCode': [team_code],
            'goalieTier': [goalie_tier]
        })
        cat_active = [c for c in self.cat_cols if c in self.active_features]
        num_active = [c for c in self.num_cols if c in self.active_features]

        df_encoded = pd.get_dummies(input_df[self.active_features], columns=cat_active, dtype=float)
        df_encoded = df_encoded.reindex(columns=self.encoded_columns, fill_value=0.0)
        df_encoded[num_active] = self.scaler.transform(df_encoded[num_active])

        return df_encoded.values.astype(np.float32)
    def transform_batch(self, df_new):
        """Transforms a new DataFrame into the format required by the models."""
        cat_active = [c for c in self.cat_cols if c in self.active_features]
        num_active = [c for c in self.num_cols if c in self.active_features]

        df_encoded = pd.get_dummies(df_new[self.active_features], columns=cat_active, dtype=float)
        df_encoded = df_encoded.reindex(columns=self.encoded_columns, fill_value=0.0)
        df_encoded[num_active] = self.scaler.transform(df_encoded[num_active])

        return df_encoded.values.astype(np.float32)