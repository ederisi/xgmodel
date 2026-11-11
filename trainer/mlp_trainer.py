from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, brier_score_loss

class MLPTrainer:
    def __init__(self, preprocessor):
        self.pre = preprocessor
        self.model = None

    def train_model(self):
        print("\nTraining MLP Neural Network xG model...")
        self.model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
        self.model.fit(self.pre.X_train, self.pre.y_train)
        
        preds = self.model.predict_proba(self.pre.X_test)[:, 1]
        loss = log_loss(self.pre.y_test, preds)
        brier = brier_score_loss(self.pre.y_test, preds)
        print(f" -> MLP Log Loss: {loss:.4f} | Brier Score: {brier:.4f}")
        return self.model

    def predict_shot(self, *args, **kwargs):
        if self.model is None:
            self.train_model()
        X_single = self.pre.transform_single(*args, **kwargs)
        return float(self.model.predict_proba(X_single)[0, 1])