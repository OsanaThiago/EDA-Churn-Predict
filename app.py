import streamlit as st
import pandas as pd
from predict import getModels
from transform.preprocessing import loading_data


@st.cache_data
def load_data():
    return loading_data()

df = load_data()

@st.cache_resource
def load_models():
    return getModels() 

model, features = load_models()

st.markdown("# Will be Churn?")

credit_score = st.sidebar.slider("Credit Score", df['CreditScore'].min(), int(df['CreditScore'].max() * 1.1))
age = st.sidebar.slider("Idade",df['Age'].min(), df["Age"].max())
tenure = st.sidebar.slider("Tenure (anos)",df['Tenure'].min(), df["Tenure"].max()+1)
balance = st.sidebar.number_input("Balance", df['Balance'].min(), df["Balance"].max() * 1.1)
estimated_salary = st.sidebar.number_input("Estimated Salary", df['EstimatedSalary'].min(), df["EstimatedSalary"].max() * 1.1)
num_products = st.sidebar.slider("Numero de Produtos Comprados",0,6)
points_earned = st.sidebar.slider("Point Earned", df['Point Earned'].min(), int(df["Point Earned"].max() * 1.1))
satisfaction_score = st.sidebar.slider("Satisfaction Score", df['Satisfaction Score'].min(), df["Satisfaction Score"].max())
geography = st.sidebar.selectbox("Selecione seu país", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Selecione seu Gênero",["Male", "Female"])
card_type = st.sidebar.selectbox("Card Type",["Silver", "Gold", "Platinum", "Diamond"])
has_cr_card = st.sidebar.selectbox("Possui Cartão de Crédito?",[0,1])
is_active_member = st.sidebar.selectbox("É Membro Ativo?",[0,1])
complain = st.sidebar.selectbox("Possui Reclamação?",[0,1])

input_data = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary,
    "Complain": complain,
    "Satisfaction Score": satisfaction_score,
    "Card Type": card_type,
    "Point Earned": points_earned
}

dados = pd.DataFrame([input_data])

probachurn = model.predict_proba(dados[features])[0,1]

st.metric("Probabilidade de dar Churn ", f"{probachurn:.2%}")