import streamlit as st
import pandas as pd
from transform.preprocessing import loading_data

df = loading_data()

@st.cache_data
def load_data():
    return df

@st.cache_data
def churn_baseline(df):
    return df["Exited"].mean()