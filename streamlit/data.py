import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    df = pd.read_csv('../data/Customer-Churn-Records.csv')
    df = df.drop(columns=["RowNumber","CustomerId","Surname",'Complain'])
    return df

@st.cache_data
def churn_baseline(df):
    return df["Exited"].mean()