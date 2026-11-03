import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import log_loss, brier_score_loss

class ShotCNN(nn.Module):
    def __init__(self, num_features):
        super(ShotCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(32 * num_features, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x


class CNNModelTrainer:
    def __init__(self, preprocessor):
        self.pre = preprocessor
        self.model = None

    def train_model(self, epochs=5, batch_size=256):
        """Trains 1D-CNN model using the preprocessed data from SklearnPreprocessor."""
        X_train = self.pre.X_train
        y_train = self.pre.y_train.astype('float32')
        X_test = self.pre.X_test
        y_test = self.pre.y_test

        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model = ShotCNN(num_features=X_train.shape[1])
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        print(f"\nTraining 1D-CNN xG model ({epochs} epochs)...")
        self.model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(X_test)).numpy().flatten()
            
        loss = log_loss(y_test, preds)
        brier = brier_score_loss(y_test, preds)
        print(f" -> 1D-CNN Log Loss: {loss:.4f} | Brier Score: {brier:.4f}")
        return self.model

    def predict_shot(self, distance, x_coord, y_coord, handedness, off_wing, team_code, goalie_tier):
        """Predicts the xG of a single shot using the trained 1D-CNN model."""
        if self.model is None:
            self.train_model()

        X_single = self.pre.transform_single(
            distance, x_coord, y_coord, handedness, off_wing, team_code, goalie_tier
        )
        
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.tensor(X_single)).item()
        return float(pred)