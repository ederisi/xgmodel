"""
New Season Data Validator Page for Streamlit App
This script defines the Streamlit application that allows users to upload a new season's MoneyPuck CSV file and compare the 
performance of 8 different machine learning models against MoneyPuck's official xG model (`xGoal`).
"""
import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import log_loss, brier_score_loss
from model_loader import get_all_models
from xwin_simul import GameWinSimulator

# Page configs
st.set_page_config(page_title="Season Data Validator", page_icon="📈", layout="wide")

# Custom CSS-style (cards, font and colors)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_validator_resources():
    """
    Load and cache all 8 models and the preprocessor for the Season Data Validator app.
    """
    xgb_trainer, cnn_trainer, lstm_trainer, gru_trainer, logreg_trainer, rf_trainer, hgb_trainer, mlp_trainer = get_all_models()
    pre = logreg_trainer.pre
    simulator = GameWinSimulator(xgb_trainer)
    return xgb_trainer, cnn_trainer, lstm_trainer, gru_trainer, logreg_trainer, rf_trainer, hgb_trainer, mlp_trainer, pre, simulator

# Load models, preprocessor, and simulator
with st.spinner("Initializing models..."):
    xgb_trainer, cnn_trainer, lstm_trainer, gru_trainer, logreg_trainer, rf_trainer, hgb_trainer, mlp_trainer, pre, simulator = load_validator_resources()

st.title("📈 New data & season validation (vs. MoneyPuck)")
st.markdown("Upload a new season's MoneyPuck CSV file and compare the performance of your **8 models + Ensemble** against MoneyPuck's official xG model (`xGoal`).")
st.markdown("---")

# File upload widget for new season CSV file
uploaded_file = st.file_uploader("📂 Upload a new season's MoneyPuck CSV file (e.g., shots_2025.csv)", type=["csv"])

if uploaded_file is not None:
    @st.cache_data
    def evaluate_uploaded_season(file_bytes):
        """
        Handles loaded data, evaluates all 8 models + Ensemble, and compares them to MoneyPuck's xGoal model.
        """
        df_new = pd.read_csv(file_bytes)
        if 'game_id' in df_new.columns and 'time' in df_new.columns:
            df_new = df_new.sort_values(['game_id', 'time']).reset_index(drop=True)
        else:
            df_new = df_new.copy()

        goal_col = 'isGoal' if ('isGoal' in df_new.columns and 'goal' not in df_new.columns) else 'goal'

        # Create a new column for goalie tiers, where goalies not in the special list are labeled as "Average Tendie"
        special_goalies = set(xgb_trainer.goalie_options) - {"Average Tendie"}
        if 'goalieNameForShot' in df_new.columns:
            goalie_tier_series = df_new['goalieNameForShot'].apply(
                lambda x: x if x in special_goalies else "Average Tendie"
            )
        else:
            goalie_tier_series = "Average Tendie"
            
        df_new = df_new.assign(goalieTier=goalie_tier_series)

        active_features = [f for f in xgb_trainer.active_features if f in df_new.columns]
        
        required_cols = active_features + [goal_col]
        has_moneypuck = 'xGoal' in df_new.columns
        if has_moneypuck:
            required_cols.append('xGoal')

        # Keep only the required columns and any relevant meta columns for game-level analysis
        meta_cols = [c for c in ['game_id', 'time', 'homeTeamCode', 'awayTeamCode', 'isHomeTeam', 'teamCode', 'shooterTeamCode'] if c in df_new.columns]
        keep_cols = list(dict.fromkeys(required_cols + meta_cols))
            
        clean_df = df_new[keep_cols].dropna(subset=required_cols).copy()
        if 'game_id' in clean_df.columns and 'time' in clean_df.columns:
            clean_df = clean_df.sort_values(['game_id', 'time']).reset_index(drop=True)
        
        # XGBoost predictions
        xgb_df = clean_df[active_features].copy()
        for col, cats in xgb_trainer.categories.items():
            if col in xgb_df.columns:
                xgb_df[col] = pd.Categorical(xgb_df[col], categories=cats)
        clean_df['xG_XGBoost'] = np.clip(xgb_trainer.model.predict_proba(xgb_df)[:, 1], 0.001, 0.999)

        # PyTorch-based DNNs (CNN, LSTM, GRU)
        X_scaled = pre.transform_batch(clean_df)
        X_seq = pre.transform_batch_sequences(clean_df)
        
        cnn_trainer.model.eval()
        with torch.no_grad():
            clean_df['xG_1D-CNN'] = np.clip(cnn_trainer.model(torch.tensor(X_scaled)).numpy().flatten(), 0.001, 0.999)

        X_seq_tensor = torch.tensor(X_seq)
        for name, trainer in [('LSTM', lstm_trainer), ('GRU', gru_trainer)]:
            trainer.model.eval()
            with torch.no_grad():
                preds = trainer.model(X_seq_tensor).numpy().flatten()
                clean_df[f'xG_{name}'] = np.clip(preds, 0.001, 0.999)

        # Sklearn models predictions
        clean_df['xG_LogReg'] = np.clip(logreg_trainer.model.predict_proba(X_scaled)[:, 1], 0.001, 0.999)
        clean_df['xG_RandomForest'] = np.clip(rf_trainer.model.predict_proba(X_scaled)[:, 1], 0.001, 0.999)
        clean_df['xG_HistGB'] = np.clip(hgb_trainer.model.predict_proba(X_scaled)[:, 1], 0.001, 0.999)
        clean_df['xG_MLP'] = np.clip(mlp_trainer.model.predict_proba(X_scaled)[:, 1], 0.001, 0.999)

        # calculate ensemble xG as the average of all 8 models
        own_models = ['xG_XGBoost', 'xG_1D-CNN', 'xG_LSTM', 'xG_GRU', 'xG_LogReg', 'xG_RandomForest', 'xG_HistGB', 'xG_MLP']
        clean_df['xG_Ensemble'] = clean_df[own_models].mean(axis=1)

        all_cols = own_models + ['xG_Ensemble']
        if has_moneypuck:
            clean_df['xG_MoneyPuck'] = np.clip(clean_df['xGoal'], 0.001, 0.999)
            all_cols.append('xG_MoneyPuck')

        return clean_df, goal_col, all_cols, has_moneypuck

    # Evaluate the uploaded season data
    with st.spinner("Evaluating new season shots across all models..."):
        clean_df, goal_col, all_model_cols, has_mp = evaluate_uploaded_season(uploaded_file)

    st.success(f"✅ Uploaded and analyzed **{len(clean_df):,}** shots! {'(MoneyPuck xGoal included)' if has_mp else ''}")

    # Model selection dropdown for detailed analysis
    default_idx = all_model_cols.index('xG_Ensemble') if 'xG_Ensemble' in all_model_cols else 0
    selected_xg_col = st.selectbox(
        "🤖 Choose a model for detailed analysis:",
        all_model_cols,
        index=default_idx
    )

    # 3 tabs for model performance, game-level xWin analysis, and team-level summary
    tab_models, tab_games, tab_teams = st.tabs([
        "📊 1. Model Performance (Accuracy & Loss)",
        "🎲 2. Game-level xWin Analysis",
        "🏆 3. Team Performance Summary"
    ])

    # Tab 1: Model performance metrics
    with tab_models:
        st.subheader("Model Performance on New Season Data")
        actual_total_goals = int(clean_df[goal_col].sum())
        y_true = clean_df[goal_col].values.astype(int)

        metrics_rows = []
        for col in all_model_cols:
            preds = np.clip(clean_df[col].values, 1e-6, 1 - 1e-6)
            pred_goals = float(preds.sum())
            metrics_rows.append({
                "Model": "⭐ " + col.replace("xG_", "") if col in ['xG_Ensemble', 'xG_MoneyPuck'] else col.replace("xG_", ""),
                "Actual Goals": actual_total_goals,
                "Predicted Total xG": round(pred_goals, 1),
                "Difference (Goals - xG)": round(actual_total_goals - pred_goals, 1),
                "Log Loss (Smaller = Better)": round(log_loss(y_true, preds), 4),
                "Brier Score (Smaller = Better)": round(brier_score_loss(y_true, preds), 4)
            })

        metrics_df = pd.DataFrame(metrics_rows).sort_values(by="Log Loss (Smaller = Better)")
        st.dataframe(metrics_df, width='stretch', hide_index=True)

    # Tab 2: Game-level xWin analysis, MoneyPuck comparison, and actual winner
    with tab_games:
        model_label = selected_xg_col.replace('xG_', '')
        st.subheader(f"Game-level xWin analysis: {model_label} vs. MoneyPuck")
        
        if 'game_id' not in clean_df.columns:
            st.warning("File does not contain 'game_id' column for game-level breakdown.")
        else:
            @st.cache_data
            def simulate_all_season_games(df_sub, xg_col, g_col, include_mp):
                """"
                Simulates all games in the season using the selected model's xG values and compares them to MoneyPuck's xG if available.
                """
                results = []
                for gid, group in df_sub.groupby('game_id'):
                    home = group['homeTeamCode'].iloc[0] if 'homeTeamCode' in group.columns else 'HOME'
                    away = group['awayTeamCode'].iloc[0] if 'awayTeamCode' in group.columns else 'AWAY'

                    if 'isHomeTeam' in group.columns:
                        h_mask = group['isHomeTeam'] == 1
                    else:
                        t_col = 'teamCode' if 'teamCode' in group.columns else 'shooterTeamCode'
                        h_mask = group[t_col] == home

                    # Chosen model's xG simulation
                    h_xg = group.loc[h_mask, xg_col].values
                    a_xg = group.loc[~h_mask, xg_col].values
                    h_goals = int(group.loc[h_mask, g_col].sum())
                    a_goals = int(group.loc[~h_mask, g_col].sum())

                    sim_own = simulator.simulate_from_xg_arrays(h_xg, a_xg, num_simulations=1000)
                    actual_winner = home if h_goals > a_goals else (away if a_goals > h_goals else "OT/SO")
                    
                    expected_winner_own = home if sim_own['home_win_prob'] >= 50.0 else away
                    correct_own = (actual_winner == expected_winner_own) if actual_winner != "OT/SO" else None

                    
                    row_data = {
                        "Game ID": gid,
                        "Game": f"{away} @ {home}",
                        "Goals (H - A)": f"{h_goals} - {a_goals}",
                        f"{model_label} xG (H - A)": f"{sim_own['home_xG']:.2f} - {sim_own['away_xG']:.2f}",
                    }
                    
                    # MoneyPuck xG simulation and comparison
                    expected_winner_mp = None
                    correct_mp = None
                    if include_mp:
                        mp_h_xg = group.loc[h_mask, 'xG_MoneyPuck'].values
                        mp_a_xg = group.loc[~h_mask, 'xG_MoneyPuck'].values
                        sim_mp = simulator.simulate_from_xg_arrays(mp_h_xg, mp_a_xg, num_simulations=1000)
                        
                        row_data["MoneyPuck xG (H - A)"] = f"{sim_mp['home_xG']:.2f} - {sim_mp['away_xG']:.2f}"
                        row_data["MoneyPuck xWin% (H - A)"] = f"{sim_mp['home_win_prob']:.1f}% - {sim_mp['away_win_prob']:.1f}%"
                        
                        expected_winner_mp = home if sim_mp['home_win_prob'] >= 50.0 else away
                        correct_mp = (actual_winner == expected_winner_mp) if actual_winner != "OT/SO" else None

                    row_data.update({
                        f"{model_label} xWin% (H - A)": f"{sim_own['home_win_prob']:.1f}% - {sim_own['away_win_prob']:.1f}%",
                        "Favorite (Model)": expected_winner_own,
                        "True Winner": actual_winner,
                        "Model Correct?": "✅ Yes" if correct_own is True else ("❌ No" if correct_own is False else "➖ OT/SO")
                    })
                    
                    if include_mp:
                        row_data["MoneyPuck Correct?"] = "✅ Yes" if correct_mp is True else ("❌ No" if correct_mp is False else "➖ OT/SO")

                    results.append(row_data)
                return pd.DataFrame(results)

            with st.spinner("Simulating games and comparing with MoneyPuck..."):
                games_df = simulate_all_season_games(clean_df, selected_xg_col, goal_col, has_mp)

            decided_games = games_df[games_df["True Winner"] != "OT/SO"]
            accuracy_own = (decided_games["Favorite (Model)"] == decided_games["True Winner"]).mean() * 100 if len(decided_games) > 0 else 0.0

            # Display summary metrics for the season's game-level analysis
            c1, c2, c3 = st.columns(3)
            c1.metric("Played matches", f"{len(games_df)} kpl")
            c2.metric(f"xWin Winner Accuracy ({model_label})", f"{accuracy_own:.1f} %")
            
            if has_mp and "MoneyPuck Correct?" in decided_games.columns and len(decided_games) > 0:
                accuracy_mp = (decided_games["MoneyPuck Correct?"].str.contains("✅")).mean() * 100
                c3.metric("xWin Winner Accuracy (MoneyPuck)", f"{accuracy_mp:.1f} %")
            else:
                c3.metric("Surprises / Upsets", f"{100 - accuracy_own:.1f} %")

            st.dataframe(games_df, width='stretch', hide_index=True)

    # Tab 3: Team Level
    with tab_teams:
        st.subheader(f"Team Performance: Actual Goals vs. {model_label} xG")
        team_col = 'teamCode' if 'teamCode' in clean_df.columns else 'shooterTeamCode'
        
        agg_dict = {
            'Shots': (goal_col, 'count'),
            'Actual Goals': (goal_col, 'sum'),
            'Model xG': (selected_xg_col, 'sum')
        }
        if has_mp:
            agg_dict['MoneyPuck xG'] = ('xG_MoneyPuck', 'sum')

        team_summary = clean_df.groupby(team_col, observed=False).agg(**agg_dict).reset_index()
        team_summary['Model xG'] = team_summary['Model xG'].round(2)
        team_summary['Goals Above Model xG'] = (team_summary['Actual Goals'] - team_summary['Model xG']).round(2)
        
        if has_mp:
            team_summary['MoneyPuck xG'] = team_summary['MoneyPuck xG'].round(2)
            team_summary['Difference (Model - MP)'] = (team_summary['Model xG'] - team_summary['MoneyPuck xG']).round(2)
            
        team_summary = team_summary.sort_values(by='Model xG', ascending=False)
        st.dataframe(team_summary, width='stretch', hide_index=True)