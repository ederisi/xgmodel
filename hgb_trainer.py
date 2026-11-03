from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, brier_score_loss

class HGBTrainer:
    def __init__(self, preprocessor):
        self.pre = preprocessor
        self.model = None

    def train_model(self):
        print("\nTraining HistGradientBoosting xG model...")
        self.model = HistGradientBoostingClassifier(max_iter=100, max_depth=5, learning_rate=0.05, random_state=42)
        self.model.fit(self.pre.X_train, self.pre.y_train)
        
        preds = self.model.predict_proba(self.pre.X_test)[:, 1]
        loss = log_loss(self.pre.y_test, preds)
        brier = brier_score_loss(self.pre.y_test, preds)
        print(f" -> HistGB Log Loss: {loss:.4f} | Brier Score: {brier:.4f}")
        return self.model

    def predict_shot(self, distance, x_coord, y_coord, handedness, off_wing, team_code, goalie_tier):
        if self.model is None:
            self.train_model()
        X_single = self.pre.transform_single(distance, x_coord, y_coord, handedness, off_wing, team_code, goalie_tier)
        return float(self.model.predict_proba(X_single)[:, 1][0])