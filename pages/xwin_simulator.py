"""
Nhl xWin Simulator Page
This Streamlit page allows users to simulate NHL game outcomes based on expected goals (xG) values for shots taken by both teams. 
Users can either select a real NHL game from the dataset or input custom team and shot parameters to run Monte Carlo simulations and obtain win probabilities.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from model_loader import get_all_models
from xwin_simul import GameWinSimulator

# Page configuration
st.set_page_config(page_title="NHL xWin Simulator", page_icon="🎲", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
""", unsafe_allow_html=True)

# Load and cache all 8 models and initialize the GameWinSimulator
with st.spinner("Initializing models and simulator..."):
    xgb_trainer, cnn_trainer, lstm_trainer, gru_trainer, logreg_trainer, rf_trainer, hgb_trainer, mlp_trainer = get_all_models()
    simulator = GameWinSimulator(xgb_trainer)

# Main title and description
st.title("🎲 NHL Expected Win (xWin%) Simulator")
st.markdown("Simulates match-specific expected win probabilities"
" **10 000 Monte Carlo -repetitions** with either real NHL game data or custom team and shot parameters.")
st.markdown("---")

# Simulation mode selection: real NHL game data or custom game simulator
mode = st.radio(
    "Select simulation mode:", 
    ["🏟️ Real NHL game data (Deserved To Win)", "⚡ Custom Game Simulator"], 
    horizontal=True
)

# Mode 1: actual match simulation based on real nhl game data
if mode == "🏟️ Real NHL game data (Deserved To Win)":
    games = simulator.get_available_games()
    if not games:
        st.warning("No 'game_id' column found in the data.")
    else:
        selected_game = st.selectbox("Select game:", games, format_func=lambda x: x[1])
        game_id = selected_game[0]

        # Run simulation for the selected game and display results
        res = simulator.simulate_real_game(game_id, num_simulations=10000)

        # display results in 3 columns: home team, away team, and distribution of simulated goals
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader(f"🏠 {res['home_team']} (Home)")
            st.metric("Win Probability (xWin%)", f"{res['home_win_prob']:.1f} %")
            st.metric("Total xG (Shots)", f"{res['home_xG']:.2f} ({res['home_shots']} shots)")
            st.metric("Actual Goals in Game", f"{res['actual_home_goals']}")
            
        with col2:
            st.subheader(f"✈️ {res['away_team']} (Away)")
            st.metric("Win Probability (xWin%)", f"{res['away_win_prob']:.1f} %")
            st.metric("Total xG (Shots)", f"{res['away_xG']:.2f} ({res['away_shots']} shots)")
            st.metric("Actual Goals in Game", f"{res['actual_away_goals']}")
            
        with col3:
            st.subheader("📊 Distribution of Simulated Goals (10 000 sims)")
            # Plot histogram of simulated goals for both teams
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            
            bins = np.arange(0, max(res['home_goals_dist'].max(), res['away_goals_dist'].max()) + 2) - 0.5
            ax.hist(res['home_goals_dist'], bins=bins, alpha=0.65, label=res['home_team'], color='#ff4b4b', density=True)
            ax.hist(res['away_goals_dist'], bins=bins, alpha=0.65, label=res['away_team'], color='#1f77b4', density=True)
            
            ax.set_xlabel("Simulated Goals", color='white')
            ax.set_ylabel("Probability", color='white')
            ax.tick_params(colors='white')
            ax.legend(facecolor='#161b22', edgecolor='none', labelcolor='white')
            st.pyplot(fig)
            st.caption(f"Probability of Overtime/Tie: **{res['ot_prob']:.1f} %**")

# Mode 2: custom game simulator with user-defined team and shot parameters
else:
    teams = sorted(list(xgb_trainer.categories.get('teamCode', ['HOME', 'AWAY'])))
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("🏠 Home Team Settings")
        h_team = st.selectbox("Home Team", teams, index=0)
        h_shots = st.slider("Home Team Shots", 5, 60, 32)
        h_avg_xg = st.slider("Average Shot Quality (xG per Shot)", 0.02, 0.25, 0.09, step=0.01)
        
    with c2:
        st.subheader("✈️ Away Team Settings")
        a_team = st.selectbox("Away Team", teams, index=min(1, len(teams)-1))
        a_shots = st.slider("Away Team Shots", 5, 60, 28)
        a_avg_xg = st.slider("Average Shot Quality (xG per Shot)", 0.02, 0.25, 0.08, step=0.01)

    # Create equal xG arrays for both teams based on the number of shots and average xG per shot
    home_xg_arr = np.full(h_shots, h_avg_xg)
    away_xg_arr = np.full(a_shots, a_avg_xg)

    #Run simulation based on the generated xG arrays and display results
    res = simulator.simulate_from_xg_arrays(home_xg_arr, away_xg_arr, num_simulations=10000)
    
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"🏠 {h_team} Win Probability", f"{res['home_win_prob']:.1f} %", f"Expected Goals: {res['home_xG']:.2f}")
    m2.metric(f"✈️ {a_team} Win Probability", f"{res['away_win_prob']:.1f} %", f"Expected Goals: {res['away_xG']:.2f}")
    m3.metric("⏱️ Overtime Probability", f"{res['ot_prob']:.1f} %")