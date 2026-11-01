import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xg_model import XGModelTrainer

st.set_page_config(page_title="Advanced NHL xG Calculator", page_icon="🏒", layout="wide")

st.title("🏒 Advanced Interactive NHL Expected Goals (xG) Calculator")
st.write("Using a Machine Learning model trained directly on MoneyPuck shot data, incorporating player handedness, off-wing angles, and specific goaltenders!")

# Initialize and load model using your trainer class (cached)
@st.cache_resource
def get_trainer():
    trainer = XGModelTrainer("shots.csv")
    trainer.train_model()
    return trainer

trainer = get_trainer()

# Extract unique lists from dataset for dropdowns
try:
    teams = sorted(trainer.df['teamCode'].unique()) if 'teamCode' in trainer.df.columns else ['TEAM A']
    goalies = sorted(trainer.df['goalieNameForShot'].dropna().unique()) if 'goalieNameForShot' in trainer.df.columns else ['Spencer']
    handedness_options = sorted(trainer.df['shooterLeftRight'].dropna().unique()) if 'shooterLeftRight' in trainer.df.columns else ['L', 'R']
except:
    teams = ['TEAM A']
    goalies = ['Spencer']
    handedness_options = ['L', 'R']

# Sidebar configuration options
st.sidebar.header("🎯 Matchup & Shot Setup")
selected_team = st.sidebar.selectbox("Select Attacking Team", teams)

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Player & Goalie Configuration")
handedness = st.sidebar.selectbox("Shooter Handedness (shooterLeftRight)", handedness_options)
goalie_choice = st.sidebar.selectbox("Guarding Goaltender", goalies)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Shot Coordinates")
x_coord = st.sidebar.slider("Distance from Centerline (X Cord)", 25.0, 89.0, 68.0)
y_coord = st.sidebar.slider("Distance from Net Center (Y Cord)", -42.0, 42.0, 0.0)

# Calculate standard distance to net (89, 0)
shot_distance = np.sqrt((89 - x_coord)**2 + y_coord**2)

# Compute off-wing flag based on dataset convention (1 if off-wing, 0 otherwise)
is_off_wing = 1 if ((handedness == "L" and y_coord > 5) or (handedness == "R" and y_coord < -5)) else 0

# Predict xG directly from the trained model using real features!
predicted_xg = trainer.predict_shot(
    distance=shot_distance,
    x_coord=x_coord,
    y_coord=y_coord,
    handedness=handedness,
    off_wing=is_off_wing,
    goalie_name=goalie_choice
)

# Layout & Visualization
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Interactive Offensive Zone Rink")
    st.write(f"Simulating shot for a **{handedness}-handed** shooter facing **{goalie_choice}**.")
    
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
    
    title_suffix = " (Off-Wing Angle Advantage!)" if is_off_wing == 1 else ""
    ax.set_title(f"Shot Map | {selected_team}{title_suffix}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Rink Length (Feet)")
    ax.set_ylabel("Rink Width (Feet)")
    ax.legend(loc='upper left')
    
    st.pyplot(fig)

with right_col:
    st.subheader("📊 Model Expected Goals (xG)")
    st.metric(label="Calculated xG", value=f"{predicted_xg:.3f}")
    
    if predicted_xg > 0.20:
        st.success("🔥 **Elite Scoring Chance!** High conversion probability.")
    elif predicted_xg > 0.08:
        st.warning("⚠️ **Good Scoring Look.** Standard opportunity.")
    else:
        st.info("❄️ **Perimeter Shot.** Low conversion probability.")
        
    st.markdown("---")
    st.write("**Data-Driven Parameters:**")
    st.markdown(f"- **Shooter Handedness:** {handedness}")
    st.markdown(f"- **Off-Wing Status:** {'Yes' if is_off_wing == 1 else 'No'}")
    st.markdown(f"- **Goaltender:** {goalie_choice}")
    st.markdown(f"- **Calculated Distance:** {shot_distance:.1f} ft")