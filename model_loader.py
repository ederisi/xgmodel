# Load and train all models once into cache to avoid retraining on every Streamlit rerun
import streamlit as st
from xg_trainer import XGModelTrainer
from cnn_trainer import CNNModelTrainer
from sklearn_pre import SklearnPreprocessor
from logreg_trainer import LogRegTrainer
from rf_trainer import RFTrainer
from hgb_trainer import HGBTrainer
from mlp_trainer import MLPTrainer

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