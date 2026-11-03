import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xg_trainer import XGModelTrainer
from cnn_trainer import CNNModelTrainer
from sklearn_pre import SklearnPreprocessor
from logreg_trainer import LogRegTrainer
from rf_trainer import RFTrainer
from hgb_trainer import HGBTrainer
from mlp_trainer import MLPTrainer

# Page configs
st.set_page_config(
    page_title="NHL xG Analytics Dashboard", 
    page_icon="🏒", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS-style (cards, font and colors)
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
    }
    .highlight-card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Cache the model training to avoid retraining on every Streamlit rerun
@st.cache_resource
def get_all_models():
    xgb = XGModelTrainer("shots.csv")
    xgb.train_model()

    pre = SklearnPreprocessor(xgb.df, xgb.active_features, xgb.goal_col)
    
    cnn = CNNModelTrainer(pre)
    cnn.train_model(epochs=5)

    logreg = LogRegTrainer(pre)
    logreg.train_model()
    
    rf = RFTrainer(pre)
    rf.train_model()
    
    hgb = HGBTrainer(pre)
    hgb.train_model()
    
    mlp = MLPTrainer(pre)
    mlp.train_model()

    return xgb, cnn, logreg, rf, hgb, mlp

with st.spinner("Loading and training 6 ML models... Wait a moment 🏒"):
    xgb_trainer, cnn_trainer, logreg_trainer, rf_trainer, hgb_trainer, mlp_trainer = get_all_models()

# title and description
st.title("🏒 NHL Expected Goals (xG) Analytics Suite")
st.markdown("Interactive tool for modeling expected goals (xG) in hockey shot situations and comparing **6 different AI models**.")
st.markdown("---")

# Sidebar configs
st.sidebar.header("⚙️ Match- and shot-specific parameters")

teams = sorted(list(xgb_trainer.categories.get('teamCode', ['TEAM A'])))
goalies = xgb_trainer.goalie_options
handedness_options = sorted(list(xgb_trainer.categories.get('shooterLeftRight', ['L', 'R'])))

selected_team = st.sidebar.selectbox("🏟️ Attacking team", teams)
handedness = st.sidebar.selectbox("✋ Shooter handedness", handedness_options)
goalie_choice = st.sidebar.selectbox("🛡️ Goalie", goalies)

st.sidebar.markdown("### 📍 Shot location on the ice")
x_coord = st.sidebar.slider("Distance from center line (X-coordinate)", 25.0, 89.0, 68.0)
y_coord = st.sidebar.slider("Distance from goal center (Y-coordinate)", -42.0, 42.0, 0.0)

shot_distance = np.sqrt((89 - x_coord)**2 + y_coord**2)
is_off_wing = 1 if ((handedness == "L" and y_coord > 5) or (handedness == "R" and y_coord < -5)) else 0

shot_args = (shot_distance, x_coord, y_coord, handedness, is_off_wing, selected_team, goalie_choice)

# Predictions from all 6 models
all_predictions = {
    "XGBoost": xgb_trainer.predict_shot(*shot_args),
    "1D-CNN": cnn_trainer.predict_shot(*shot_args),
    "Logistic Regression": logreg_trainer.predict_shot(*shot_args),
    "Random Forest": rf_trainer.predict_shot(*shot_args),
    "HistGradientBoosting": hgb_trainer.predict_shot(*shot_args),
    "MLP Neural Net": mlp_trainer.predict_shot(*shot_args)
}
avg_xg = np.mean(list(all_predictions.values()))

# Main formatted output with tabs
tab1, tab2 = st.tabs(["🎯 Shot Map & Prediction", "🏆 Team Statistics (Leaderboard)"])

with tab1:
    col_rink, col_metrics = st.columns([1.5, 1])
    
    with col_rink:
        st.subheader("Attacking zone visual map")
        st.write(f"Shot from team **{selected_team}** ({'Off-Wing' if is_off_wing else 'Normal'} angle).")
        
        # Clean and modern sports analytics style chart
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_xlim(25, 100)
        ax.set_ylim(-45, 45)
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        
        # Lines and goalie box
        ax.axvline(x=89, color='#ff4b4b', linewidth=2.5, label='Goal line')
        ax.axvline(x=54, color='#1f77b4', linewidth=1.5, linestyle='--', label='Center line')
        net_box = plt.Rectangle((89, -3), 3, 6, edgecolor='white', facecolor='#262730', linewidth=1.5, zorder=3)
        ax.add_patch(net_box)
        
        # Shot location
        marker_color = '#b83280' if is_off_wing == 1 else '#ff4b4b'
        ax.scatter([x_coord], [y_coord], color=marker_color, s=250, zorder=5, edgecolor='white', linewidth=1.5, label='Laukaus')
        
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.legend(loc='upper left', facecolor='#161b22', edgecolor='none', labelcolor='white')
        
        st.pyplot(fig)

    with col_metrics:
        st.subheader("📊 Expected Goals (xG)")
        
        # Main metric for average xG across all models
        st.metric(
            label="⭐ Ensemble Average (6 models)", 
            value=f"{avg_xg:.3f}",
            delta="High danger situation" if avg_xg > 0.15 else "Normal situation"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("**Model-specific results:**")
        
        # Models in two clean columns
        m_cols = st.columns(2)
        for idx, (model_name, xg_val) in enumerate(all_predictions.items()):
            with m_cols[idx % 2]:
                diff = xg_val - avg_xg
                st.metric(label=model_name, value=f"{xg_val:.3f}", delta=f"{diff:+.3f}")

with tab2:
    st.subheader("Team Statistics (Leaderboard)")
    st.write("Based on the sum of all shots' expected goals (xG) values.")
    
    with st.spinner("Calculating team statistics..."):
        leaderboard = xgb_trainer.get_team_leaderboard()
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)