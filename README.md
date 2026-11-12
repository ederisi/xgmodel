# 🏒 NHL Expected Goals (xG) & Expected Win (xWin) Analytics Suite

Modular ML and Dnn system for NHL's expected goal (xG), expected win% (xWin%) and goalie save above expected (GSAX) modelling. Application includes an interactive streamlit-UI and compares eight different models with Moneypuck's own xGoal/xWin model.

---

## ✨ Features

* **🤖 8 ML/DL-model architecture + Ensemble:**
  * **Treebased & Gradient Boosting -models:** `XGBoost`, `HistGradientBoosting`, `Random Forest`.
  * **Timeseries- ja deep learning models (PyTorch):** `1D-CNN` (convolutional network with BatchNorm- ja Logit-calibration) and sequencebased `Sequence LSTM`- ja `Sequence GRU`-models, which analyze the shot's as a chain with a length of four (`seq_len=4`).
  * **Classical and neural network models (Scikit-Learn):** `Logistic Regression`, `MLP Neural Network`.
  * **Log-Odds Bias -calibrating:** PyTorch-models outputs are calibrated automatically to respond to training data's true goal average.
* **⚡ Feature Engineering:**
  * Utilizes shot distance, coordinates (`X`, `Y`), shot angle (`shotAngleAdjusted`), handedness (`offWing`), type (`shotType`), rebounds (`shotRebound`), rushes (`shotRush`), team and goalie tier(`goalieTier`).
* **🎯 Interactive shot map & Situational analysis (`app.py`):**
  * Move shot location and game situation in real time in the offensive zone map. Compare the 8 model's given xG-values and situational impact (Rush/Rebound).
* **🎲 Monte Carlo xWin -match simulator (`pages/xwin_simulator.py` & `xwin_simul.py`):**
  * Simulates match outcome 10 000 times based on single shot xG-values and calculates the expected win probability for both teams (`xWin%`).
* **📈 New season validation vs. MoneyPuck (`pages/season_valid.py`):**
  * Loads new season's data (fe. `shots_2026.csv`) and compares model accuracy (**Log Loss**, **Brier Score**, **Predicted Total xG** ja **xWin Winner Accuracy**) to moneypuck's model.
* **🥅 Goalie's GSAX-analyysi (`pages/goalie_gsax.py`):**
  * Computes goalie specific **Goals Saved Above Expected (GSAX = faced xG − Actual goals)** -stats with chosen goalie and filters results based on actual faced shots.

---

## 📁 Project structure

```text
xgmodel/
│
├── app.py                        # Main Page: Interactive shot map, 8 model xG comparison & Leaderboard
├── main.py                       # Run script for the streamlit server
├── model_loader.py               # Initializes, trains and caches (@st.cache_resource) all 8 models
├── xwin_simul.py                 # Monte Carlo -match simulator computation logic
│
├── pages/                        # Streamlit's multi tab analysis views
│   ├── goalie_gsax.py            # Goalie GSAX-analysis ja Top/Bottom-visualisation
│   ├── season_valid.py           # New season's csv validation (Log Loss, Brier Score & xWin vs. MoneyPuck)
│   └── xwin_simulator.py         # Match specific xwin simulator, with input parameter tuning
│
├── trainer/                      # Modular ML and Dnn models
│   ├── __init__.py               # Packet initialization
│   ├── xg_trainer.py             # XGBoost-model, data loading and data preprocessing
│   ├── sklearn_pre.py            # StandardScaler, One-Hot encoding & 3D-sequence (seq_len=4) building
│   ├── cnn_trainer.py            # PyTorch 1D-CNN -model + Log-Odds-calibration
│   ├── lstm_trainer.py           # PyTorch True Sequence LSTM -model
│   ├── gru_trainer.py            # PyTorch True Sequence GRU -model
│   ├── hgb_trainer.py            # Scikit-Learn HistGradientBoosting -model
│   ├── rf_trainer.py             # Scikit-Learn Random Forest -model
│   ├── logreg_trainer.py         # Scikit-Learn Logistic Regression -model
│   └── mlp_trainer.py            # Scikit-Learn Multi-Layer Perceptron (MLP) -neural network
│
├── shots.csv                     # Training data (MoneyPuck NHL shot data)
├── shots_2026.csv                # Test-/Validation seasons shot data
└── team_x_goals_summary.csv      # Team level summary
```

## How to setup
- Ensure you have Python 3.10+ version
- download all needed libraries (with command): pip install streamlit pandas numpy scikit-learn xgboost torch matplotlib
- Load data, I utilized Moneypuck's shot data. Place this inside the project.
- If you want to utilize another season's data for validation, you can upload it straight to either the season_val or goalie_gsax tab

## How to run
- Either by py(can be python or similar depends on your pc configurations) py main.py or streamlit app.py
- (The app opens on your browser)
- The system evaluates the models based on log loss and brier score. The models located in the trainer file,
- to find suitable performance.
