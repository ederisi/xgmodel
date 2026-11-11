import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import log_loss, brier_score_loss

class ShotLSTM(nn.Module):
    def __init__(self, num_features):
        super(ShotLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=num_features, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2
        )
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x on is now a true shot sequence of shape (batch_size, seq_len, num_features)
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]  # Take the output of the last time step
        out = self.relu(self.fc1(last_step))
        out = self.dropout(out)
        return self.sigmoid(self.fc2(out))


class LSTMTrainer:
    def __init__(self, preprocessor):
        self.pre = preprocessor
        self.model = None

    def train_model(self, epochs=8, batch_size=256):
        X_train = self.pre.X_seq_train
        y_train = self.pre.y_train.astype('float32')
        X_test = self.pre.X_seq_test
        y_test = self.pre.y_test

        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model = ShotLSTM(num_features=X_train.shape[2])
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        print(f"\nTraining True Sequence LSTM xG model ({epochs} epochs, seq_len={self.pre.seq_len})...")
        self.model.train()
        for _ in range(epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()
        self.model.eval()
        with torch.no_grad():
            train_preds = self.model(torch.tensor(X_train)).numpy().flatten()
            mean_pred = float(np.clip(train_preds.mean(), 1e-5, 1 - 1e-5))
            target_mean = float(y_train.mean())
            logit_shift = np.log(target_mean / (1.0 - target_mean)) - np.log(mean_pred / (1.0 - mean_pred))
            self.model.fc2.bias.data += float(logit_shift)

            preds = self.model(torch.tensor(X_test)).numpy().flatten()
            
        print(f" -> Sequence LSTM Log Loss: {log_loss(y_test, preds):.4f} | Brier Score: {brier_score_loss(y_test, preds):.4f}")
        return self.model

    def predict_shot(self, *args, **kwargs):
        if self.model is None:
            self.train_model()

        X_single = self.pre.transform_single(*args, **kwargs)
        X_seq = np.repeat(X_single[:, np.newaxis, :], self.pre.seq_len, axis=1)

        self.model.eval()
        with torch.no_grad():
            return float(self.model(torch.tensor(X_seq)).item())