import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_csv("../data/Customer-Churn-Records.csv")


@st.cache_data
def churn_baseline(df):
    return df["Exited"].mean()