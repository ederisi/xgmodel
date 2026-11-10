import subprocess
import sys

def main():
    print("=== Launching NHL xG & xWin Analytics Dashboard ===")
    print("[i] Models will initialize and cache inside the Streamlit server.")
    print("[i] Press Ctrl + C in this terminal whenever you want to shut it down.\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n[+] Streamlit server stopped cleanly. See you next time!")
    except Exception as e:
        print(f"Error launching Streamlit app: {e}")

if __name__ == "__main__":
    main()