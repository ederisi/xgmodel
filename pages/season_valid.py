# New page for validating new season data against MoneyPuck's xG model and our own models.
import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import log_loss, brier_score_loss
from model_loader import get_all_models
from xwin_simul import GameWinSimulator

st.set_page_config(page_title="Season Data Validator", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
""", unsafe_allow_html=True)

with st.spinner("Initializing models..."):
    xgb_trainer, cnn_trainer, logreg_trainer, rf_trainer, hgb_trainer, mlp_trainer = get_all_models()
    pre = logreg_trainer.pre
    simulator = GameWinSimulator(xgb_trainer)

st.title("📈 New Data & Season Validation (vs. MoneyPuck)")
st.markdown("Upload MoneyPuck's new season CSV file and compare your **6 models + Ensemble** performance directly against **MoneyPuck's official xG model** (`xGoal`).")
st.markdown("---")

uploaded_file = st.file_uploader("📂 Upload new season MoneyPuck CSV file (e.g., shots_2025.csv)", type=["csv"])

if uploaded_file is not None:
    @st.cache_data
    def evaluate_uploaded_season(file_bytes):
        df_new = pd.read_csv(file_bytes)
        goal_col = 'isGoal' if ('isGoal' in df_new.columns and 'goal' not in df_new.columns) else 'goal'
        

        # Connect the top and bottom goalie tiers from training data to the new season data
        special_goalies = set(xgb_trainer.goalie_options) - {"Average Tendie"}
        if 'goalieNameForShot' in df_new.columns:
            df_new['goalieTier'] = df_new['goalieNameForShot'].apply(
                lambda x: x if x in special_goalies else "Average Tendie"
            )
        else:
            df_new['goalieTier'] = "Average Tendie"

        active_features = [f for f in xgb_trainer.active_features if f in df_new.columns]
        
        # If MoneyPuck's xGoal column is present, include it in the required columns for validation
        required_cols = active_features + [goal_col]
        has_moneypuck = 'xGoal' in df_new.columns
        if has_moneypuck:
            required_cols.append('xGoal')
            
        clean_df = df_new.dropna(subset=required_cols).copy()

        # XG predictions for all models
        xgb_df = clean_df[active_features].copy()
        for col, cats in xgb_trainer.categories.items():
            if col in xgb_df.columns:
                xgb_df[col] = pd.Categorical(xgb_df[col], categories=cats)
        clean_df['xG_XGBoost'] = xgb_trainer.model.predict_proba(xgb_df)[:, 1]

        # Scaled model inputs for CNN, LogReg, RandomForest, HistGB, and MLP
        X_scaled = pre.transform_batch(clean_df)
        
        cnn_trainer.model.eval()
        with torch.no_grad():
            clean_df['xG_1D-CNN'] = cnn_trainer.model(torch.tensor(X_scaled)).numpy().flatten()
            
        clean_df['xG_LogReg'] = logreg_trainer.model.predict_proba(X_scaled)[:, 1]
        clean_df['xG_RandomForest'] = rf_trainer.model.predict_proba(X_scaled)[:, 1]
        clean_df['xG_HistGB'] = hgb_trainer.model.predict_proba(X_scaled)[:, 1]
        clean_df['xG_MLP'] = mlp_trainer.model.predict_proba(X_scaled)[:, 1]

        own_models = ['xG_XGBoost', 'xG_1D-CNN', 'xG_LogReg', 'xG_RandomForest', 'xG_HistGB', 'xG_MLP']
        clean_df['xG_Ensemble'] = clean_df[own_models].mean(axis=1)

        all_cols = own_models + ['xG_Ensemble']
        if has_moneypuck:
            clean_df['xG_MoneyPuck'] = clean_df['xGoal']
            all_cols.append('xG_MoneyPuck')

        return clean_df, goal_col, all_cols, has_moneypuck

    with st.spinner("Evaluating new season shots across all models..."):
        clean_df, goal_col, all_model_cols, has_mp = evaluate_uploaded_season(uploaded_file)

    st.success(f"✅ Loaded and analyzed **{len(clean_df):,}** shots! {'(MoneyPuck xGoal included)' if has_mp else ''}")

    # Presumption: If MoneyPuck's xGoal is present, it will be included in the comparison metrics and visualizations.
    default_idx = all_model_cols.index('xG_Ensemble') if 'xG_Ensemble' in all_model_cols else 0
    selected_xg_col = st.selectbox(
        "🤖 Select your model for detailed game and team-level comparison:",
        all_model_cols,
        index=default_idx
    )

    tab_models, tab_games, tab_teams = st.tabs([
        "📊 1. Model Accuracy Comparison (Own vs. MoneyPuck)",
        "🎲 2. Game-level xWin% vs. Actual Winner",
        "🏆 3. Team Performance"
    ])

    # Tab 1: All model's + MoneyPuck's xG performance metrics
    with tab_models:
        st.subheader("Model Performance on New Season Data")
        actual_total_goals = int(clean_df[goal_col].sum())
        y_true = clean_df[goal_col].values

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

    # Tab 2: Game-level xWin analysis with MoneyPuck comparison and actual winner
    with tab_games:
        model_label = selected_xg_col.replace('xG_', '')
        st.subheader(f"Game-level xWin analysis ({model_label} vs. Actual Result)")
        
        if 'game_id' not in clean_df.columns:
            st.warning("File does not contain a 'game_id' column for game-level analysis.")
        else:
            @st.cache_data
            def simulate_all_season_games(df_sub, xg_col, g_col, include_mp):
                results = []
                for gid, group in df_sub.groupby('game_id'):
                    home = group['homeTeamCode'].iloc[0] if 'homeTeamCode' in group.columns else 'HOME'
                    away = group['awayTeamCode'].iloc[0] if 'awayTeamCode' in group.columns else 'AWAY'

                    if 'isHomeTeam' in group.columns:
                        h_mask = group['isHomeTeam'] == 1
                    else:
                        t_col = 'teamCode' if 'teamCode' in group.columns else 'shooterTeamCode'
                        h_mask = group[t_col] == home

                    h_xg = group.loc[h_mask, xg_col].values
                    a_xg = group.loc[~h_mask, xg_col].values
                    h_goals = int(group.loc[h_mask, g_col].sum())
                    a_goals = int(group.loc[~h_mask, g_col].sum())

                    sim = simulator.simulate_from_xg_arrays(h_xg, a_xg, num_simulations=2500)

                    actual_winner = home if h_goals > a_goals else (away if a_goals > h_goals else "TIE/OT")
                    expected_winner = home if sim['home_win_prob'] >= 50.0 else away
                    correct_call = (actual_winner == expected_winner) if actual_winner != "TIE/OT" else None

                    row_data = {
                        "Game ID": gid,
                        "Game": f"{away} @ {home}",
                        "Goals (H - A)": f"{h_goals} - {a_goals}",
                        "Predicted xG (H - A)": f"{sim['home_xG']:.2f} - {sim['away_xG']:.2f}",
                    }
                    
                    if include_mp:
                        mp_h = float(group.loc[h_mask, 'xG_MoneyPuck'].sum())
                        mp_a = float(group.loc[~h_mask, 'xG_MoneyPuck'].sum())
                        row_data["MoneyPuck xG (H - A)"] = f"{mp_h:.2f} - {mp_a:.2f}"

                    row_data.update({
                        "Home xWin %": round(sim['home_win_prob'], 1),
                        "Away xWin %": round(sim['away_win_prob'], 1),
                        "xWin favorite": expected_winner,
                        "True winner": actual_winner,
                        "xWin favorite correct?": "✅ Yes" if correct_call is True else ("❌ No (Surprise)" if correct_call is False else "➖ TIE/OT")
                    })
                    results.append(row_data)
                return pd.DataFrame(results)

            with st.spinner("Simulate all matches (xWin%)..."):
                games_df = simulate_all_season_games(clean_df, selected_xg_col, goal_col, has_mp)

            decided_games = games_df[games_df["True winner"] != "TIE/OT"]
            accuracy = (decided_games["xWin favorite"] == decided_games["True winner"]).mean() * 100 if len(decided_games) > 0 else 0.0

            c1, c2, c3 = st.columns(3)
            c1.metric("played matches in data", f"{len(games_df)} kpl")
            c2.metric(f"xWin-favorite won ({model_label})", f"{accuracy:.1f} %")
            c3.metric("Surprises / Goalie Saves", f"{100 - accuracy:.1f} %")

            st.dataframe(games_df, use_container_width=True, hide_index=True)

    # tab 3: team level (own models vs. MoneyPuck vs. goals) with difference column
    with tab_teams:
        model_label = selected_xg_col.replace('xG_', '')
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
        st.dataframe(team_summary, use_container_width=True, hide_index=True)