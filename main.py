import subprocess
import sys
from sklearn_pre import SklearnPreprocessor
from xg_trainer import XGModelTrainer
from cnn_trainer import CNNModelTrainer
from logreg_trainer import LogRegTrainer
from rf_trainer import RFTrainer
from hgb_trainer import HGBTrainer
from mlp_trainer import MLPTrainer

def main():
    print("=== Initializing 6-Model NHL xG System ===")
    
    # Train xGBoost model and load data
    xgb_trainer = XGModelTrainer(data_path="shots.csv")
    xgb_trainer.train_model()

    # Preprocess data for sklearn models
    pre = SklearnPreprocessor(xgb_trainer.df, xgb_trainer.active_features, goal_col=xgb_trainer.goal_col)
    
    # Train 1D-CNN
    cnn_trainer = CNNModelTrainer(pre)
    cnn_trainer.train_model(epochs=5)


    # Train Logistic Regression model
    logreg_trainer = LogRegTrainer(pre)
    logreg_trainer.train_model()

    # Train Random Forest model
    rf_trainer = RFTrainer(pre)
    rf_trainer.train_model()

    # Train HistGradientBoosting model
    hgb_trainer = HGBTrainer(pre)
    hgb_trainer.train_model()

    # Train Multi-Layer Perceptron model
    mlp_trainer = MLPTrainer(pre)
    mlp_trainer.train_model()
    
    print("\n[+] All 6 models trained. Launching Streamlit GUI Application...")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n[+] Streamlit server stopped cleanly. See you next time!")
    except Exception as e:
        print(f"Error launching Streamlit app: {e}")

if __name__ == "__main__":
    main()