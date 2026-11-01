import subprocess
import sys
from xg_trainer import XGModelTrainer
from cnn_trainer import CNNModelTrainer

def main():
    print("=== Initializing NHL xG System ===")
    
    # Train xGBoost model
    trainer = XGModelTrainer(data_path="shots.csv")
    trainer.train_model()
    
    # Train 1D-CNN model utilizing the same DataFrame and features as the XGBoost model
    cnn_trainer = CNNModelTrainer(
        df=trainer.df, 
        active_features=trainer.active_features, 
        goal_col=trainer.goal_col
    )
    cnn_trainer.train_model(epochs=5)
    
    # Print team stats to console
    print("\n--- Generating Console Team Leaderboard Summary (XGBoost) ---")
    leaderboard = trainer.get_team_leaderboard()
    print(leaderboard.head(10).to_string(index=False))
    
    print("\n[+] Backend training complete. Launching Streamlit GUI Application...")
    print("[i] The app is running. Press Ctrl + C in this terminal whenever you want to shut it down.\n")
    
    # Start streamlit app
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n[+] Streamlit server stopped cleanly. See you next time!")
    except Exception as e:
        print(f"Error launching Streamlit app: {e}")

if __name__ == "__main__":
    main()