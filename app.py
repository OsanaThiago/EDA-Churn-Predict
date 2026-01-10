import streamlit as st
import pandas as pd
import shap
from predict import getModels
from transform.preprocessing import loading_data

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")

@st.cache_data
def load_data():
    return loading_data()

@st.cache_resource
def load_models():
    return getModels()

df = load_data()
pipeline, features = load_models()

preprocess = pipeline.named_steps["preprocessar"]
model = pipeline.named_steps["model"].best_estimator_
feature_names = preprocess.get_feature_names_out().tolist()
feature_names = [f.replace("num__","").replace("cat__","").replace("_",": ") for f in feature_names]

st.title("📉 Predição de Churn de Clientes")
st.markdown("Ferramenta de apoio à **decisão comercial**, com explicabilidade do modelo.")

st.sidebar.header("Dados do Cliente")

credit_score = st.sidebar.slider("Credit Score", int(df["CreditScore"].min()), int(df["CreditScore"].max()))
age = st.sidebar.slider("Idade", int(df["Age"].min()), int(df["Age"].max()))
tenure = st.sidebar.slider("Tempo como cliente (anos)", int(df["Tenure"].min()), int(df["Tenure"].max()))
balance = st.sidebar.number_input("Saldo em Conta", 0.0)
estimated_salary = st.sidebar.number_input("Salário Estimado", 0.0)
num_products = st.sidebar.slider("Número de Produtos", 0, 6)
points_earned = st.sidebar.slider("Pontos Acumulados", int(df["Point Earned"].min()), int(df["Point Earned"].max()))
satisfaction_score = st.sidebar.slider("Satisfaction Score", int(df["Satisfaction Score"].min()), int(df["Satisfaction Score"].max()))
geography = st.sidebar.selectbox("País",["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gênero",["Male", "Female"])
card_type = st.sidebar.selectbox("Tipo de Cartão",["Silver", "Gold", "Platinum", "Diamond"])
has_cr_card = st.sidebar.selectbox("Possui Cartão de Crédito?",["Não", "Sim"])
is_active_member = st.sidebar.selectbox("Membro Ativo?",["Não", "Sim"])

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

if probachurn < 0.30:
    risco = "🟢 Baixo Risco"
elif probachurn < 0.60:
    risco = "🟡 Médio Risco"
else:
    risco = "🔴 Alto Risco"

col1, col2 = st.columns(2)

with col1:
    st.metric("Probabilidade de Churn", f"{probachurn:.2%}")

with col2:
    st.metric("Classificação de Risco", risco)

st.progress(min(int(probachurn * 100), 100))
st.markdown("---")
st.subheader("Por que esse cliente pode dar churn?")

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

features_transformed = preprocess.transform(dados[features])
explainer = load_explainer(model)
shap_values = explainer.shap_values(features_transformed)

shap_df = pd.DataFrame({"Feature": feature_names, "Impacto": shap_values[0]})
shap_df["Impacto Absoluto"] = shap_df["Impacto"].abs()
shap_df = shap_df.sort_values("Impacto Absoluto", ascending=False).head(6)
shap_df["Efeito no Churn"] = shap_df["Impacto"].apply(lambda v: "Aumenta risco" if v > 0 else "Reduz risco")

st.dataframe(shap_df[["Feature", "Efeito no Churn"]], use_container_width=True, hide_index=True)

st.subheader("💡 Principais fatores de risco")

for _, row in shap_df.iterrows(): # tentar percorrer o df['impacto'] em vez de iterrows
    if row["Impacto"] > 0:
        st.write(f"🔴**{row['Feature']}** está aumentando o risco de churn")
    else:
        st.write(f"🟢 **{row['Feature']}** ajuda na retenção do cliente")

st.subheader("Recomendações Comerciais")

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

st.markdown("---")
st.subheader("Resumo Executivo")

st.markdown(f""" 
            Este cliente possui **{probachurn:.1%} de probabilidade de churn**,
            classificado como **{risco}**. Os principais fatores que influenciam esse risco estão relacionados a:
            **{', '.join(shap_df['Feature'].head(3))}**.
            """)
