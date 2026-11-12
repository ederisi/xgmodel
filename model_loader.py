"""
Model loader that aggregates all 8 models and returns them as a tuple. This is used in the main Streamlit app to load and cache the models for predictions.
This file is also utilized as cache handler. Module is responsible for loading and caching
all 8 models (XGBoost, CNN, LSTM, GRU, Logistic Regression, Random Forest, HistGradientBoosting, MLP) for the NHL xG & xWin Analytics Dashboard.
"""
import streamlit as st
from trainer.xg_trainer import XGModelTrainer
from trainer.cnn_trainer import CNNModelTrainer
from trainer.lstm_trainer import LSTMTrainer
from trainer.gru_trainer import GRUTrainer
from trainer.sklearn_pre import SklearnPreprocessor
from trainer.logreg_trainer import LogRegTrainer
from trainer.rf_trainer import RFTrainer
from trainer.hgb_trainer import HGBTrainer
from trainer.mlp_trainer import MLPTrainer

@st.cache_resource
def get_all_models():
    """
    Load and cache all 8 models for the NHL xG & xWin Analytics Dashboard.
    Returns: Tuple of all trained models.
    """

    # xgboost model and data preprocessing
    xgb = XGModelTrainer("shots.csv")
    xgb.train_model()

    # Create a mutual sklearn preprocessor (scaling, one-hot and sequences)
    pre = SklearnPreprocessor(xgb.df, xgb.active_features, xgb.goal_col)

    cnn = CNNModelTrainer(pre)
    cnn.train_model(epochs=10)

    logreg = LogRegTrainer(pre)
    logreg.train_model()

    rf = RFTrainer(pre)
    rf.train_model()
    
    hgb = HGBTrainer(pre)
    hgb.train_model()
    
    mlp = MLPTrainer(pre)
    mlp.train_model()
    
    lstm = LSTMTrainer(pre)
    lstm.train_model(epochs=10)
    
    gru = GRUTrainer(pre)
    gru.train_model(epochs=10)

    # return all models as a tuple
    return xgb, cnn, lstm, gru, logreg, rf, hgb, mlp