# Load and train all models once into cache to avoid retraining on every Streamlit rerun
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
    xgb = XGModelTrainer("shots.csv")
    xgb.train_model()
    
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
    
    return xgb, cnn, lstm, gru, logreg, rf, hgb, mlp