import pandas as pd
import numpy as np
from xgboost import XGBClassifier

print("Loading shot data for team analysis...")
df = pd.read_csv("shots_2025.csv")

# Identify key columns
goal_col = 'goal' if 'goal' in df.columns else 'isGoal'

# MoneyPuck usually uses 'teamCode' for the team taking the shot
team_col = None
for col in ['teamCode', 'shooterTeamCode', 'team']:
    if col in df.columns:
        team_col = col
        break

if not team_col:
    print("Could not find team column. Here are available columns:", [c for c in df.columns if 'team' in c.lower()])
    exit()

print(f"Using team column: '{team_col}' and goal column: '{goal_col}'")

# Features for our xG model
features = ['arenaAdjustedShotDistance', 'arenaAdjustedXCordABS', 'arenaAdjustedYCordAbs']
df = df.dropna(subset=features + [goal_col, team_col])

# Train model on the data to score every shot
X = df[features]
y = df[goal_col]

print("Training xG model across all shots...")
model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.05, 
    max_depth=4, 
    random_state=42,
    eval_metric='logloss'
)
model.fit(X, y)

# Predict xG for every shot
df['xG'] = model.predict_proba(X)[:, 1]

# Aggregate stats by team
print("\nCalculating team-level totals...")
team_stats = df.groupby(team_col).agg(
    Games=('game_id', 'nunique') if 'game_id' in df.columns else ('arenaAdjustedShotDistance', 'count'), # fallback count if game_id missing
    Total_Shots=(goal_col, 'count'),
    Actual_Goals=(goal_col, 'sum'),
    Total_xG=('xG', 'sum')
).reset_index()

# Calculate finishing luck/performance (Actual Goals minus Expected Goals)
team_stats['Goals_Above_Expected'] = team_stats['Actual_Goals'] - team_stats['Total_xG']

# Round decimals for clean viewing
team_stats['Total_xG'] = team_stats['Total_xG'].round(2)
team_stats['Goals_Above_Expected'] = team_stats['Goals_Above_Expected'].round(2)

# Sort by Total xG descending
team_stats = team_stats.sort_values(by='Total_xG', ascending=False)

print("\n--- NHL Team xG Leaderboard ---")
print(team_stats.to_string(index=False))