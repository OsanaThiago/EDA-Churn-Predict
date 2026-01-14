import shap
import pandas as pd

class ShapService:
    def __init__(self, model, preprocess, feature_names):
        self.model = model
        self.preprocess = preprocess
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, dados):
        features_transformed = self.preprocess.transform(dados)
        shap_values = self.explainer.shap_values(features_transformed)

        shap_df = pd.DataFrame({"Feature": self.feature_names, "Impacto": shap_values[0]})

        shap_df["Impacto Absoluto"] = shap_df["Impacto"].abs()
        shap_df = shap_df.sort_values("Impacto Absoluto", ascending=False).head(6)
        shap_df["Efeito no Churn"] = shap_df["Impacto"].apply(lambda v: "Aumenta risco" if v > 0 else "Reduz risco")

        return shap_df
