import streamlit as st
import pandas as pd
import numpy as np
import os

import plotly.express as px
from dashboard.kpi import kpis
from dashboard.main_chart import main_chart
from dashboard.subplots import gender_age_heatmap, risk_flags_chart, bmi_diabetes_chart

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(layout="wide")


def prepare_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "EDA", "transformed_for_eda.csv"))
    df["diabetes_label"] = df["diabetes"].map({0: "No Diabetes", 1: "Diabetes"})
    return df


def main():
    tab_dashboard, tab_interactive = st.tabs(["Dashboard", "Predictor"])
    
    with tab_dashboard:
        st.header("Diabetes Predictor Dashboard")
        df = prepare_data()
        kpis(df)
        st.subheader("Feature Distribution Analysis")
        main_chart(df)
        gender_age_heatmap(df)
        col1, col2 = st.columns([0.55, 0.45])
        with col1:
            risk_flags_chart(df)
        with col2:
            bmi_diabetes_chart(df)
            
    with tab_interactive:
        st.header("Interactive Form")
        
        
        


main()
