import streamlit as st
import pandas as pd
from pathlib import Path

@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "Customer-Churn-Records.csv"
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["RowNumber","CustomerId","Surname",'Complain'])
    return df

@st.cache_data
def churn_baseline(df):
    return df["Exited"].mean()