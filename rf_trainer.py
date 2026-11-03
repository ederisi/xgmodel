from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, brier_score_loss

class RFTrainer:
    def __init__(self, preprocessor):
        self.pre = preprocessor
        self.model = None

    def train_model(self):
        print("\nTraining Random Forest xG model...")
        self.model = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1, random_state=42)
        self.model.fit(self.pre.X_train, self.pre.y_train)
        
        preds = self.model.predict_proba(self.pre.X_test)[:, 1]
        loss = log_loss(self.pre.y_test, preds)
        brier = brier_score_loss(self.pre.y_test, preds)
        print(f" -> Random Forest Log Loss: {loss:.4f} | Brier Score: {brier:.4f}")
        return self.model

    def predict_shot(self, distance, x_coord, y_coord, handedness, off_wing, team_code, goalie_tier):
        if self.model is None:
            self.train_model()
        X_single = self.pre.transform_single(distance, x_coord, y_coord, handedness, off_wing, team_code, goalie_tier)
        return float(self.model.predict_proba(X_single)[:, 1][0])