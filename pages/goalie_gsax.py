"""
Goalie GSAX (Goals Saved Above Expected) Analytics Streamlit App
This script defines the Streamlit application that provides an interactive interface for analyzing goalie performance in hockey shot situations 
using multiple machine learning models. The app calculates the GSAX metric, which measures a goalie's performance relative to expected goals (xG).
"""
import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from model_loader import get_all_models

# Page configs
st.set_page_config(page_title="Goalie GSAX Analytics", page_icon="🥅", layout="wide")

# Custom CSS-style (cards, font and colors)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_goalie_resources():
    """
    Load and cache all 8 models and the preprocessor for the Goalie GSAX Analytics app.
    """
    xgb, cnn, lstm, gru, logreg, rf, hgb, mlp = get_all_models()
    return xgb, cnn, lstm, gru, logreg, rf, hgb, mlp, logreg.pre

# Load models and preprocessor
xgb_trainer, cnn_trainer, lstm_trainer, gru_trainer, logreg_trainer, rf_trainer, hgb_trainer, mlp_trainer, pre = load_goalie_resources()

# Main title and description
st.title("🥅 NHL Goalie GSAX (Goals Saved Above Expected) Analytics")
st.markdown("Analyze goalies' performance relative to expected goals (**GSAX = Observed xG − Goals Allowed**). A positive GSAX indicates the goalie has saved more goals than expected.")
st.markdown("---")

# Choose data source: either the default training data or upload a new season CSV file
data_source = st.radio(
    "Choose data source:",
    ["📂 Training data (shots.csv)", "⬆️ Upload new season CSV file"],
    horizontal=True
)

uploaded_csv = None
if data_source == "⬆️ Upload new season CSV file":
    uploaded_csv = st.file_uploader("Select MoneyPuck CSV file", type=["csv"])

@st.cache_data
def compute_goalie_predictions(file_bytes=None):
    """
    Compute xG predictions for goalies based on the provided CSV file or the default training data.
    """
    df = pd.read_csv(file_bytes) if file_bytes is not None else xgb_trainer.df.copy()
    goal_col = 'isGoal' if ('isGoal' in df.columns and 'goal' not in df.columns) else 'goal'

    # Reorder dataframe by game_id and time if those columns exist
    if 'game_id' in df.columns and 'time' in df.columns:
        df = df.sort_values(['game_id', 'time']).reset_index(drop=True)

    # Evaluate goalies objectively (compared to"Average Tendie"),
    # in order for the GSAX to compare to average goalie
    df['goalieTier'] = "Average Tendie"
    active_features = [f for f in xgb_trainer.active_features if f in df.columns]
    
    req_cols = active_features + [goal_col, 'goalieNameForShot']
    has_mp = 'xGoal' in df.columns
    if has_mp:
        req_cols.append('xGoal')

    clean_df = df.dropna(subset=req_cols).copy()

    # XGBoost
    xgb_df = clean_df[active_features].copy()
    for col, cats in xgb_trainer.categories.items():
        if col in xgb_df.columns:
            xgb_df[col] = pd.Categorical(xgb_df[col], categories=cats)
    clean_df['xG_XGBoost'] = np.clip(xgb_trainer.model.predict_proba(xgb_df)[:, 1], 0.001, 0.999)

    # 2D-models (CNN + Sklearn)
    X_scaled = pre.transform_batch(clean_df)
    cnn_trainer.model.eval()
    with torch.no_grad():
        clean_df['xG_1D-CNN'] = np.clip(cnn_trainer.model(torch.tensor(X_scaled)).numpy().flatten(), 0.001, 0.999)

    # 3D-sequencemodels (LSTM & GRU)
    X_seq = pre.transform_batch_sequences(clean_df)
    X_seq_tensor = torch.tensor(X_seq)
    for name, trainer in [('LSTM', lstm_trainer), ('GRU', gru_trainer)]:
        trainer.model.eval()
        with torch.no_grad():
            clean_df[f'xG_{name}'] = np.clip(trainer.model(X_seq_tensor).numpy().flatten(), 0.001, 0.999)

    # Sklearn models
    clean_df['xG_LogReg'] = np.clip(logreg_trainer.model.predict_proba(X_scaled)[:, 1], 0.001, 0.999)
    clean_df['xG_RandomForest'] = np.clip(rf_trainer.model.predict_proba(X_scaled)[:, 1], 0.001, 0.999)
    clean_df['xG_HistGB'] = np.clip(hgb_trainer.model.predict_proba(X_scaled)[:, 1], 0.001, 0.999)
    clean_df['xG_MLP'] = np.clip(mlp_trainer.model.predict_proba(X_scaled)[:, 1], 0.001, 0.999)

    # Calculate ensemble xG as the average of all 8 models
    own_cols = ['xG_XGBoost', 'xG_1D-CNN', 'xG_LSTM', 'xG_GRU', 'xG_LogReg', 'xG_RandomForest', 'xG_HistGB', 'xG_MLP']
    clean_df['xG_Ensemble'] = clean_df[own_cols].mean(axis=1)

    model_list = own_cols + ['xG_Ensemble']
    if has_mp:
        clean_df['xG_MoneyPuck'] = clean_df['xGoal']
        model_list.append('xG_MoneyPuck')

    return clean_df, goal_col, model_list, has_mp

# Compute if the data source is the default training data or if a new CSV file is uploaded
if data_source == "📂 Training data (shots.csv)" or uploaded_csv is not None:
    with st.spinner("Calculating goalies' GSAX statistics..."):
        df_goalies, goal_col, model_cols, has_mp = compute_goalie_predictions(uploaded_csv)

    # Sidebar controls for model selection and minimum shots filter
    c_ctrl1, c_ctrl2 = st.columns(2)
    with c_ctrl1:
        default_idx = model_cols.index('xG_Ensemble') if 'xG_Ensemble' in model_cols else 0
        selected_model = st.selectbox("🤖 Choose model for GSAX calculation:", model_cols, index=default_idx)
    with c_ctrl2:
        min_shots = st.slider("🎯 Minimum number of shots (filter):", 50, 1500, 300, step=50)

    # Group by goalie and compute GSAX metrics
    agg_dict = {
        'Shots_Faced': (goal_col, 'count'),
        'Goals_Allowed': (goal_col, 'sum'),
        'Expected_Goals (xG)': (selected_model, 'sum')
    }
    if has_mp:
        agg_dict['MoneyPuck_xG'] = ('xG_MoneyPuck', 'sum')

    g_df = df_goalies.groupby('goalieNameForShot', observed=False).agg(**agg_dict).reset_index()
    g_df = g_df[g_df['Shots_Faced'] >= min_shots].copy()

    # Compute additional metrics: Save percentage, GSAX, and GSAX per 100 shots
    g_df['Save_%'] = ((1.0 - g_df['Goals_Allowed'] / g_df['Shots_Faced']) * 100).round(2)
    g_df['Expected_Goals (xG)'] = g_df['Expected_Goals (xG)'].round(2)
    g_df['GSAX (Model)'] = (g_df['Expected_Goals (xG)'] - g_df['Goals_Allowed']).round(2)
    g_df['GSAX / 100 Shots'] = ((g_df['GSAX (Model)'] / g_df['Shots_Faced']) * 100).round(2)

    if has_mp:
        g_df['GSAX (MoneyPuck)'] = (g_df['MoneyPuck_xG'] - g_df['Goals_Allowed']).round(2)

    g_df = g_df.sort_values(by='GSAX (Model)', ascending=False).reset_index(drop=True)

    # Display top and bottom goalies with metrics
    if len(g_df) >= 2:
        best_g = g_df.iloc[0]
        worst_g = g_df.iloc[-1]
        m1, m2, m3 = st.columns(3)
        m1.metric("🥇 Best GSAX-goalie", f"{best_g['goalieNameForShot']}", f"+{best_g['GSAX (Model)']:.2f} GSAX")
        m2.metric("🥶 Weakest GSAX-goalie", f"{worst_g['goalieNameForShot']}", f"{worst_g['GSAX (Model)']:.2f} GSAX")
        m3.metric("Filtered goalies", f"{len(g_df)} kpl", f"Min. {min_shots} laukausta")

    st.subheader(f"📊 Goalie GSAX-Board ({selected_model.replace('xG_', '')})")
    st.dataframe(g_df, width='stretch', hide_index=True)

    # Visual Top 10 vs Bottom 10 GSAX -chart
    st.subheader("📈 Top 10 & Bottom 10 Goalies (GSAX)")
    top_bot_df = pd.concat([g_df.head(10), g_df.tail(10)]).drop_duplicates().sort_values(by='GSAX (Model)', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    colors = ['#2ecc71' if v >= 0 else '#ff4b4b' for v in top_bot_df['GSAX (Model)']]
    ax.barh(top_bot_df['goalieNameForShot'], top_bot_df['GSAX (Model)'], color=colors)
    ax.axvline(0, color='white', linewidth=1, linestyle='--')
    ax.set_xlabel("GSAX (Goals Saved Above Expected)", color='white')
    ax.tick_params(colors='white')
    st.pyplot(fig)