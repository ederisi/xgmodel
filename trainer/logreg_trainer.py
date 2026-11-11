from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

class LogRegTrainer:
    def __init__(self, preprocessor):
        self.pre = preprocessor
        self.model = None

    def train_model(self):
        print("\nTraining Logistic Regression xG model...")
        self.model = LogisticRegression(max_iter=500, random_state=42)
        self.model.fit(self.pre.X_train, self.pre.y_train)
        
        preds = self.model.predict_proba(self.pre.X_test)[:, 1]
        loss = log_loss(self.pre.y_test, preds)
        brier = brier_score_loss(self.pre.y_test, preds)
        print(f" -> LogReg Log Loss: {loss:.4f} | Brier Score: {brier:.4f}")
        return self.model

    def predict_shot(self, *args, **kwargs):
        if self.model is None:
            self.train_model()
        X_single = self.pre.transform_single(*args, **kwargs)
        return float(self.model.predict_proba(X_single)[0, 1])