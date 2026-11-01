import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xg_trainer import XGModelTrainer
from cnn_trainer import CNNModelTrainer

st.set_page_config(page_title="Advanced NHL xG Calculator", page_icon="🏒", layout="wide")

st.title("🏒 Advanced Interactive NHL Expected Goals (xG) Calculator")
st.write("Model considers: shot location, shooter handedness, attacking team performance, and goaltender tier (Top 5 / Bottom 5 / Average Tendie)!")

@st.cache_resource
def get_models():
    xgb_trainer = XGModelTrainer("shots.csv")
    xgb_trainer.train_model()
    
    cnn_trainer = CNNModelTrainer(xgb_trainer.df, xgb_trainer.active_features, xgb_trainer.goal_col)
    cnn_trainer.train_model(epochs=5)
    return xgb_trainer, cnn_trainer

trainer, cnn_trainer = get_models()

teams = sorted(list(trainer.categories.get('teamCode', ['TEAM A'])))
goalies = trainer.goalie_options
handedness_options = sorted(list(trainer.categories.get('shooterLeftRight', ['L', 'R'])))

# Sidebar for user inputs
st.sidebar.header("🎯 Matchup & Shot Setup")
selected_team = st.sidebar.selectbox("Select Attacking Team", teams)

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Player & Goalie Configuration")
handedness = st.sidebar.selectbox("Shooter Handedness", handedness_options)
goalie_choice = st.sidebar.selectbox("Guarding Goaltender (Top 5 / Bottom 5 / Average)", goalies)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Shot Coordinates")
x_coord = st.sidebar.slider("Distance from Centerline (X Cord)", 25.0, 89.0, 68.0)
y_coord = st.sidebar.slider("Distance from Net Center (Y Cord)", -42.0, 42.0, 0.0)

shot_distance = np.sqrt((89 - x_coord)**2 + y_coord**2)
is_off_wing = 1 if ((handedness == "L" and y_coord > 5) or (handedness == "R" and y_coord < -5)) else 0

# Predict xG using the trained xgboost model
predicted_xg = trainer.predict_shot(
    distance=shot_distance,
    x_coord=x_coord,
    y_coord=y_coord,
    handedness=handedness,
    off_wing=is_off_wing,
    team_code=selected_team,
    goalie_tier=goalie_choice
)
# Predict xG using the trained CNN model
cnn_predicted_xg = cnn_trainer.predict_shot(
    distance=shot_distance,
    x_coord=x_coord,
    y_coord=y_coord,
    handedness=handedness,
    off_wing=is_off_wing,
    team_code=selected_team,
    goalie_tier=goalie_choice
)

left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Interactive Offensive Zone Rink")
    st.write(f"**{selected_team}** ({handedness}-handed) vs. **{goalie_choice}**.")
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(25, 100)
    ax.set_ylim(-45, 45)
    ax.set_facecolor('#f4f6f9')
    
    ax.axvline(x=89, color='red', linewidth=2, label='Goal Line')
    ax.axvline(x=54, color='blue', linewidth=1.5, linestyle='--', label='Blue Line')
    
    net_box = plt.Rectangle((89, -3), 3, 6, edgecolor='black', facecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(net_box)
    
    marker_color = 'purple' if is_off_wing == 1 else 'crimson'
    ax.scatter([x_coord], [y_coord], color=marker_color, s=220, zorder=5, edgecolor='black', linewidth=1.5, label=f'Shot ({selected_team})')
    
    title_suffix = " (Off-Wing Advantage!)" if is_off_wing == 1 else ""
    ax.set_title(f"Shot Map | {selected_team}{title_suffix}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Rink Length (Feet)")
    ax.set_ylabel("Rink Width (Feet)")
    ax.legend(loc='upper left')
    
    st.pyplot(fig)

with right_col:
    st.subheader("📊 Model Comparison (xG)")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric(label="XGBoost xG", value=f"{predicted_xg:.3f}")
    with m_col2:
        diff = cnn_predicted_xg - predicted_xg
        st.metric(label="1D-CNN xG", value=f"{cnn_predicted_xg:.3f}", delta=f"{diff:+.3f} vs XGB")
        
    st.markdown("---")
    st.write("**Data-Driven Parameters:**")
    st.markdown(f"- **Attacking Team:** {selected_team}")
    st.markdown(f"- **Shooter Handedness:** {handedness}")
    st.markdown(f"- **Off-Wing Status:** {'Yes' if is_off_wing == 1 else 'No'}")
    st.markdown(f"- **Goaltender Tier:** {goalie_choice}")
    st.markdown(f"- **Calculated Distance:** {shot_distance:.1f} ft")