import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

# Define 1D-CNN-neural network architecture for xG prediction
class ShotCNN(nn.Module):
    def __init__(self, num_features):
        super(ShotCNN, self).__init__()
        # 1D Convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        
        # Fully connected layers for decision making
        self.fc1 = nn.Linear(32 * num_features, 64)
        self.dropout = nn.Dropout(0.2) # Stops overfitting
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid() # Changes output to probability (0-1)

    def forward(self, x):
        # Format input: (batch_size, Channels=1, feature amount)
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x


class CNNModelTrainer:
    def __init__(self, df, active_features, goal_col='goal'):
        # Utilize the already handled DataFrame from XGModelTrainer to
        # avoid redundant data loading and preprocessing.
        self.df = df
        self.active_features = active_features
        self.goal_col = goal_col
        self.model = None
        self.scaler = StandardScaler()
        self.encoded_columns = None
        self.num_cols = ['arenaAdjustedShotDistance', 'arenaAdjustedXCordABS', 'arenaAdjustedYCordAbs', 'offWing']
        self.cat_cols = ['shooterLeftRight', 'teamCode', 'goalieTier']

    def _preprocess(self, df_subset, is_training=True):
        """Converts categories to One-Hot encoding and scales numeric values."""
        df_encoded = pd.get_dummies(df_subset[self.active_features], columns=[c for c in self.cat_cols if c in self.active_features], dtype=float)
        
        if is_training:
            self.encoded_columns = df_encoded.columns
            df_encoded[self.num_cols] = self.scaler.fit_transform(df_encoded[self.num_cols])
        else:
            # Ensure the same columns are present during prediction and scale numeric features
            df_encoded = df_encoded.reindex(columns=self.encoded_columns, fill_value=0.0)
            df_encoded[self.num_cols] = self.scaler.transform(df_encoded[self.num_cols])
            
        return df_encoded.values.astype(np.float32)

    def train_model(self, epochs=5, batch_size=256):
        """Trains the 1D-CNN model and calculates Log Loss and Brier Score."""
        clean_df = self.df.dropna(subset=self.active_features + [self.goal_col]).copy()
        
        X = self._preprocess(clean_df, is_training=True)
        y = clean_df[self.goal_col].values.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model = ShotCNN(num_features=X.shape[1])
        criterion = nn.BCELoss() # Binary Cross-Entropy  (same as log loss)
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

        # Evaluation with test data
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(X_test)).numpy().flatten()
            
        loss = log_loss(y_test, preds)
        brier = brier_score_loss(y_test, preds)
        
        print(f"CNN Model Trained Successfully!")
        print(f" -> CNN Log Loss: {loss:.4f} | CNN Brier Score: {brier:.4f}")
        return self.model

    def predict_shot(self, distance, x_coord, y_coord, handedness, off_wing, team_code, goalie_tier):
        """Predicts the expected goals (xG) for a single shot based on the provided parameters."""
        if self.model is None:
            self.train_model()

        input_df = pd.DataFrame({
            'arenaAdjustedShotDistance': [distance],
            'arenaAdjustedXCordABS': [x_coord],
            'arenaAdjustedYCordAbs': [abs(y_coord)],
            'shooterLeftRight': [handedness],
            'offWing': [off_wing],
            'teamCode': [team_code],
            'goalieTier': [goalie_tier]
        })

        X_Single = self._preprocess(input_df, is_training=False)
        
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.tensor(X_Single)).item()
        return float(pred)