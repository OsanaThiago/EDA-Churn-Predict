import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
    
import streamlit as st
import pandas as pd
from churnmodel import ChurnModel 
from shapservice import ShapService 
from recomendar import Recomendar 
from data import load_data, churn_baseline
from predict import getModels

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")
st.write("Inicializando aplicação...")

st.write("Inicializando data...")
@st.cache_data
def loading_data():
    return load_data()
df = loading_data()
baseline = churn_baseline(df)

st.write("Inicializando model...")
@st.cache_resource
def loading_model():
    return getModels()
pipeline, features = loading_model()
st.write("Inicializando features...")
@st.cache_resource
def load_modelfeature():
    preprocess = pipeline.named_steps["preprocessar"]
    feature_names = preprocess.get_feature_names_out().tolist()
    feature_names = [f.replace("num__", "").replace("cat__", "").replace("_", ": ") for f in feature_names]
    model = pipeline.named_steps["model"].best_estimator_
    return preprocess, model, feature_names

preprocess, model, feature_names = load_modelfeature()
st.write("Inicializando services...")
@st.cache_resource
def load_services(_pipeline, _model, _preprocess, feature_names, df, features):
    modelservice = ChurnModel(pipeline, features, df)
    shapservice = ShapService(model, preprocess, feature_names)
    recoservice = Recomendar()
    return modelservice, shapservice, recoservice

modelservice, shapservice, recoservice = load_services(pipeline, model, preprocess, feature_names, df, features)

st.title("📉 Predição de Churn de Clientes")
st.markdown("Ferramenta de apoio à **decisão comercial**, com explicabilidade do modelo.")

st.sidebar.header("Dados do Cliente")

credit_score = st.sidebar.slider("Credit Score", int(df["CreditScore"].min()), int(df["CreditScore"].max()))
age = st.sidebar.slider("Idade", int(df["Age"].min()), int(df["Age"].max()))
tenure = st.sidebar.slider("Tempo como cliente (anos)", int(df["Tenure"].min()), int(df["Tenure"].max()))
balance = st.sidebar.number_input("Saldo em Conta", 0.0)
estimated_salary = st.sidebar.number_input("Salário Estimado", 0.0)
num_products = st.sidebar.slider("Número de Produtos", 0, 4)
points_earned = st.sidebar.slider("Pontos Acumulados", int(df["Point Earned"].min()), int(df["Point Earned"].max()))
satisfaction_score = st.sidebar.slider("Satisfaction Score", int(df["Satisfaction Score"].min()), int(df["Satisfaction Score"].max()))
geography = st.sidebar.selectbox("País", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gênero", ["Male", "Female"])
card_type = st.sidebar.selectbox("Tipo de Cartão", ["Silver", "Gold", "Platinum", "Diamond"])
has_cr_card = st.sidebar.selectbox("Possui Cartão de Crédito?", ["Não", "Sim"])
is_active_member = st.sidebar.selectbox("Membro Ativo?", ["Não", "Sim"])

user_info = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": 1 if has_cr_card == "Sim" else 0,
    "IsActiveMember": 1 if is_active_member == "Sim" else 0,
    "EstimatedSalary": estimated_salary,
    "Satisfaction Score": satisfaction_score,
    "Card Type": card_type,
    "Point Earned": points_earned
}

dados = pd.DataFrame([user_info])
probachurn = pipeline.predict_proba(dados[features])[0, 1]
percentil_risco = modelservice.percentil_risco(probachurn)

if probachurn < 0.30:
    risco = "🟢 Baixo Risco"
elif probachurn < 0.65:
    risco = "🟡 Médio Risco"
else:
    risco = "🔴 Alto Risco"

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Probabilidade de Churn", f"{probachurn:.2%}", delta=f"{(probachurn - baseline):+.2%} vs média da base")

with col2:
    st.metric("Classificação de Risco", risco)

with col3: 
    st.metric("Posição de Risco", f"Top {(1 - percentil_risco):.0%} da base")

st.progress(min(int(probachurn * 100), 100))
st.markdown("---")
st.subheader("Por que esse cliente pode dar churn?")

shap_df = shapservice.explain(dados)

st.markdown(f"Os principais fatores de risco estão relacionados a **{', '.join(shap_df[shap_df['Impacto'] > 0]['Feature'][:3])}**.")
st.caption("Abaixo estão os fatores que mais influenciaram a previsão para este cliente específico.")

st.dataframe(shap_df[["Feature", "Efeito no Churn"]], width='stretch', hide_index=True)

st.subheader("Principais fatores de risco")

for _, row in shap_df.iterrows():
    if row["Impacto"] > 0:
        st.write(f"🔴 **{row['Feature']}** está aumentando o risco de churn")
    else:
        st.write(f"🟢 **{row['Feature']}** ajuda na retenção do cliente")

st.markdown("---")
st.subheader("Recomendações Comerciais")

recomendacoes = recoservice.recomendacao(probachurn,shap_df)

if probachurn > 0.60:
    st.error("🔴 Cliente com ALTO risco de churn")
    st.write("- Contato imediato")
    st.write("- Oferta personalizada")
    st.write("- Revisão de tarifas / benefícios")
elif probachurn > 0.30:
    st.warning("🟡 Cliente com risco MODERADO")
    st.write("- Campanha de engajamento")
    st.write("- Oferta de upgrade ou pontos bônus")
else:
    st.success("🟢 Cliente com BAIXO risco")
    st.write("- Manter relacionamento")
    st.write("- Cross-sell de produtos")

for rec in recomendacoes:
    st.write(f'- {rec}')

st.markdown("---")
st.subheader("Resumo Executivo")

st.markdown(f""" 
            Este cliente possui **{probachurn:.1%} de probabilidade de churn**,
            classificado como **{risco}**. Os principais fatores que influenciam esse risco estão relacionados a:
            **{', '.join(shap_df['Feature'].head(3))}**.
            """)
