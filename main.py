import subprocess
import sys
from xg_model import XGModelTrainer

def main():
    print("=== Initializing NHL xG System ===")
    
    # Run the backend pipeline to verify training and print team stats to the console
    trainer = XGModelTrainer(data_path="shots.csv")
    trainer.train_model()
    
    print("\n--- Generating Console Team Leaderboard Summary ---")
    leaderboard = trainer.get_team_leaderboard()
    print(leaderboard.head(10).to_string(index=False))
    
    print("\n[+] Backend training complete. Launching Streamlit GUI Application...")
    
    # Call app.py via Streamlit using subprocess
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n[+] Streamlit server stopped cleanly. See you next time!")
    except Exception as e:
        print(f"Error launching Streamlit app: {e}")

if __name__ == "__main__":
    main()