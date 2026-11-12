"""
Main Launcher for the NHL xG & xWin Analytics Dashboard.
This script serves as the entry point for launching the Streamlit application.
"""
import subprocess
import sys

def main():
    # Launch the Streamlit app in a subprocess
    print("=== Launching NHL xG & xWin Analytics Dashboard ===")
    print("[i] Models will initialize and cache inside the Streamlit server.")
    print("[i] Press Ctrl + C in this terminal whenever you want to shut it down.\n")
    
    try:
        # Execute the Streamlit command to run the app.py script
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        # Handle graceful shutdown on Ctrl + C
        print("\n[!] Streamlit server is shutting down...")
        print("\n[+] Streamlit server stopped cleanly. See you next time!")
    except Exception as e:
        # Handle any other exceptions that may occur
        print(f"[!] An error occurred while launching the Streamlit app: {e}")

if __name__ == "__main__":
    main()